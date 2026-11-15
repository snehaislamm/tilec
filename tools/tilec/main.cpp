// tilec driver.
#include "tilec/IR.h"
#include "tilec/Interpreter.h"
#include "tilec/LLVMCodegen.h"
#include "tilec/Lexer.h"
#include "tilec/Parser.h"
#include "tilec/Passes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tilec;

namespace {

const char *kUsage = R"(usage: tilec <input.tl> [options]

Output:
  --emit=<what>          ir      tile IR straight from the frontend
                         ir-opt  tile IR after the optimization pipeline
                         llvm    LLVM IR (default)
                         header  C prototypes for the generated kernels
  -o <file>              write output to <file> (default: stdout)

Pipeline:
  -O0                    skip canonicalize / cse / dce
  --no-fusion            keep one loop nest per op
  --recompute-budget=N   max body cost duplicated into multiple consumers
                         (default 4, 0 = never recompute)
  --no-tile              naive i-j-k matmul instead of cache-blocked tiles
  --tile=M,N,K           default matmul tile sizes (a kernel's @tile wins)

Debugging:
  --print-ir-after-all   dump IR to stderr after every pass
  --verify-each          run the IR verifier after every pass
  --time-passes          per-pass wall time and op counts on stderr
  --stats                kernel counts before/after fusion on stderr

Reference interpreter:
  --interp --kernel <name> --in <a.bin> [--in <b.bin> ...] --out <o.bin>
         [--stage=input|opt]   interpret frontend IR or optimized IR (default opt)
         Tensors are raw little-endian float32, row-major.
)";

std::string readFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void reportError(const std::string &path, const std::string &src, const CompileError &e) {
  std::cerr << path << ":" << e.loc.line << ":" << e.loc.col << ": error: " << e.what() << "\n";
  std::istringstream ss(src);
  std::string line;
  for (int i = 0; i < e.loc.line && std::getline(ss, line); ++i) {
  }
  if (e.loc.line > 0) {
    std::cerr << "  " << line << "\n  " << std::string(std::max(0, e.loc.col - 1), ' ') << "^\n";
  }
}

bool startsWith(const std::string &s, const std::string &p) {
  return s.compare(0, p.size(), p) == 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string input, output, emit = "llvm", kernel, stage = "opt", outBin;
  std::vector<std::string> inBins;
  PassOptions popts;
  CodegenOptions copts;
  bool interp = false, stats = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "tilec: " << flag << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      std::cout << kUsage;
      return 0;
    } else if (a == "--version") {
      std::cout << "tilec 0.1.0\n";
      return 0;
    } else if (startsWith(a, "--emit=")) {
      emit = a.substr(7);
    } else if (a == "-o") {
      output = value("-o");
    } else if (a == "-O0") {
      popts.optimize = false;
    } else if (a == "--no-fusion") {
      popts.fusion = false;
    } else if (startsWith(a, "--recompute-budget=")) {
      popts.recomputeBudget = std::atoi(a.c_str() + 19);
    } else if (a == "--no-tile") {
      copts.tileMatmul = false;
    } else if (startsWith(a, "--tile=")) {
      long long m, n, k;
      if (std::sscanf(a.c_str() + 7, "%lld,%lld,%lld", &m, &n, &k) != 3 || m <= 0 || n <= 0 ||
          k <= 0) {
        std::cerr << "tilec: --tile expects three positive integers M,N,K\n";
        return 2;
      }
      copts.tileM = m;
      copts.tileN = n;
      copts.tileK = k;
    } else if (a == "--print-ir-after-all") {
      popts.printAfterAll = true;
    } else if (a == "--verify-each") {
      popts.verifyEach = true;
    } else if (a == "--time-passes") {
      popts.timePasses = true;
    } else if (a == "--stats") {
      stats = true;
    } else if (a == "--interp") {
      interp = true;
    } else if (a == "--kernel") {
      kernel = value("--kernel");
    } else if (startsWith(a, "--stage=")) {
      stage = a.substr(8);
    } else if (a == "--in") {
      inBins.push_back(value("--in"));
    } else if (a == "--out") {
      outBin = value("--out");
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "tilec: unknown option '" << a << "'\n";
      return 2;
    } else if (input.empty()) {
      input = a;
    } else {
      std::cerr << "tilec: multiple input files\n";
      return 2;
    }
  }
  if (input.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  if (emit != "ir" && emit != "ir-opt" && emit != "llvm" && emit != "header") {
    std::cerr << "tilec: unknown --emit kind '" << emit << "'\n";
    return 2;
  }

  std::string src;
  try {
    src = readFile(input);
  } catch (const std::exception &e) {
    std::cerr << "tilec: " << e.what() << "\n";
    return 1;
  }

  IRModule mod;
  try {
    mod = buildIR(parse(lex(src)));
  } catch (const CompileError &e) {
    reportError(input, src, e);
    return 1;
  }

  try {
    std::vector<int> kernelsBefore;
    for (Function &f : mod.funcs) {
      Function tmp = f;
      outline(tmp);
      dce(tmp);
      kernelsBefore.push_back(countKernels(tmp));
    }

    bool runPipeline = !(emit == "ir" || (interp && stage == "input"));
    if (runPipeline) {
      for (size_t fi = 0; fi < mod.funcs.size(); ++fi) {
        Function &f = mod.funcs[fi];
        PassManager pm(popts, std::cerr);
        pm.run(f);
        if (popts.timePasses) {
          std::cerr << "// pass timing for @" << f.name << "\n";
          double total = 0;
          for (const PassStat &s : pm.stats()) {
            std::cerr << "//   " << std::left << std::setw(14) << s.name << std::right << std::fixed
                      << std::setprecision(3) << std::setw(9) << s.millis << " ms  ops "
                      << s.opsBefore << " -> " << s.opsAfter << (s.changed ? "  (changed)" : "")
                      << "\n";
            total += s.millis;
          }
          std::cerr << "//   total " << std::fixed << std::setprecision(3) << total << " ms\n";
        }
        if (stats)
          std::cerr << "// @" << f.name << ": " << kernelsBefore[fi] << " kernel(s) unfused -> "
                    << countKernels(f) << " kernel(s) after pipeline\n";
      }
    }

    if (interp) {
      const Function *fn = nullptr;
      for (const Function &f : mod.funcs)
        if (f.name == kernel || (kernel.empty() && mod.funcs.size() == 1)) fn = &f;
      if (!fn) {
        std::cerr << "tilec: --interp needs --kernel <name> naming a kernel in the file\n";
        return 2;
      }
      if (outBin.empty()) {
        std::cerr << "tilec: --interp needs --out <file>\n";
        return 2;
      }
      std::vector<Tensor> args;
      for (int i = 0; i < fn->numArgs(); ++i) {
        if (i >= static_cast<int>(inBins.size())) {
          std::cerr << "tilec: kernel @" << fn->name << " takes " << fn->numArgs() << " inputs\n";
          return 2;
        }
        std::string raw = readFile(inBins[i]);
        Tensor t;
        t.shape = fn->ops[i].shape;
        if (raw.size() != static_cast<size_t>(t.shape.numel()) * 4) {
          std::cerr << "tilec: " << inBins[i] << " has " << raw.size() / 4 << " floats, expected "
                    << t.shape.numel() << "\n";
          return 1;
        }
        t.data.resize(t.shape.numel());
        std::memcpy(t.data.data(), raw.data(), raw.size());
        args.push_back(std::move(t));
      }
      Tensor r = interpret(*fn, args);
      std::ofstream o(outBin, std::ios::binary);
      o.write(reinterpret_cast<const char *>(r.data.data()),
              static_cast<std::streamsize>(r.data.size() * 4));
      return o ? 0 : 1;
    }

    std::ostringstream out;
    if (emit == "ir" || emit == "ir-opt")
      print(mod, out);
    else if (emit == "header")
      emitHeader(mod, out);
    else
      emitLLVM(mod, copts, out);

    if (output.empty() || output == "-") {
      std::cout << out.str();
    } else {
      std::ofstream o(output);
      o << out.str();
      if (!o) {
        std::cerr << "tilec: cannot write '" << output << "'\n";
        return 1;
      }
    }
  } catch (const std::logic_error &e) {
    std::cerr << "tilec: internal compiler error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception &e) {
    std::cerr << "tilec: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
