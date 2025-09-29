import argparse
import os
import time
from expmanager import *

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=str)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--asan", action="store_true")
    parser.add_argument("--ablation", type=str, default="default")
    parser.add_argument("--vs", type=str, default=None)
    parser.add_argument("--apis", nargs='*', type=str, default=None)
    parser.add_argument("--time_budget", type=int, default=None)
    parser.add_argument("--rep", type=int, default=1)

    # gcov mode only
    parser.add_argument("--itv", type=int, default=None)
    parser.add_argument("--itv_total", type=int, default=None)
    parser.add_argument("--gen_html", action="store_true")

    # Resources
    parser.add_argument("--cpu_capacity", type=int, default=None)
    parser.add_argument("--mem", type=int, default=16)

    args = parser.parse_args()

    check(args.dll in ["torch", "tf"],
          f"Error: Invalid target library `{args.dll}`. Choose from [torch, tf].")

    if args.vs != None and args.apis != None:
        check(False,
            "Error: Please specify one of `--vs` or `--apis` options, not both.")

    baselines = ["acetest", "deeprel", "freefuzz", "titanfuzz", "ivysyn"]
    if args.vs != None and args.vs not in baselines:
        check(False,
            f"Error: Invalid baseline `{args.vs}`. Please specify one of {baselines}.")
    if args.vs == "ivysyn" and args.dll != "torch":
        check(False,
            f"Error: Invalid DL library `{args.dll}`. We target PyTorch kernels only for comparison with IvySyn.") 

    check(args.mode in ["fuzz", "gcov"],
          f"Error: Invalid mode `{args.mode}`. Choose from [fuzz, gcov].")
          
    check(args.ablation in ["default", "wo_staged", "wo_nbp"],
          f"Error: Invalid ablation option `{args.ablation}`. Choose from [default, wo_staged, wo_nbp].")

    if args.time_budget is None:
        if args.dll == "tf" and not args.asan and args.vs != None:
            default_time_budget = 3600
            print(emphasize(f"Note: `--time_budget` is not specified. Use default time budget for RQ1 TensorFlow, {default_time_budget}sec."))
            args.time_budget = default_time_budget
        else:
            default_time_budget = 1200
            print(emphasize(f"Note: `--time_budget` is not specified. Use default time budget {default_time_budget}sec."))
            args.time_budget = default_time_budget

    if args.mode == "gcov":
        if args.itv_total is None:
            args.itv_total = args.time_budget
        if args.itv is None:
            args.itv = args.itv_total

    if args.cpu_capacity is None:
        full_cpu = os.cpu_count()
        half_cpu = full_cpu // 2
        print(emphasize(f"Note: `--cpu_capacity` is not specified. Use half of available CPUs ({half_cpu} out of {full_cpu})."))
        args.cpu_capacity = half_cpu

    return args

def main():
    start_time = time.time()

    args = parse_arg()

    if args.dll == "torch":
        if args.version is None:
            if args.vs != None and args.vs == "ivysyn":
                args.version = "1.11"
            else:
                args.version = "2.2"
            print(emphasize(f"Note: `--version` is not specified. Use default setting `{args.version}`."))
        dll_info = TorchInfo(args.version)
    elif args.dll == "tf":
        if args.version is None:
            args.version = "2.16"
            print(emphasize(f"Note: `--version` is not specified. Use default setting `{args.version}`."))
        dll_info = TFInfo(args.version) 

    manager = ExpManager(dll_info=dll_info,
                         mode=args.mode,
                         asan=args.asan,
                         ablation=args.ablation,
                         vs=args.vs,
                         apis=args.apis,
                         time_budget=args.time_budget,
                         rep=args.rep,
                         itv=args.itv,
                         itv_total=args.itv_total,
                         gen_html=args.gen_html,
                         cpu_capacity=args.cpu_capacity,
                         mem=args.mem)

    generations = manager.schedule()
    JobRunner(args.cpu_capacity, generations).run()

    if args.mode == "gcov":
        manager.summary_gcov()
    elif args.mode == "fuzz" and args.asan:
        manager.gen_pov()

    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")

if __name__ == "__main__":
    main()
