import os
import sys
import shutil
import subprocess
from pathlib import Path
import time
import argparse

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--cmd_prefix",
    type=str,
  )
  parser.add_argument(
    "--target_dir",
    type=str,
  )
  parser.add_argument(
    "--third_party_dir",
    type=str,
    default=None,
  )
  parser.add_argument(
    "--itv_mode",
    type=str,
    help="Interval mode. one of {time, gen}.",
  )
  parser.add_argument(
    "--itv_start",
    type=int,
  )
  parser.add_argument(
    "--itv_end",
    type=int,
  )
  parser.add_argument(
    "--apis",
    type=str,
    nargs='+',
  )
  parser.add_argument(
    "--out",
    type=str,
  )
  return parser.parse_args()


def home():
  return Path.home()

def cov_tool_path():
  return os.path.join(home(), "coverage.py")

def cov_result_dir():
  return os.path.join(home(), "coverage_result")

def fuzz_result_dir():
  return os.path.join(home(), "experiment_result")

def api_result_dir(api):
  return os.path.join(fuzz_result_dir(), api)

def api_corpus_dir(api):
  return os.path.join(api_result_dir(api), "corpus")


class Proc:
  def __init__(self, cmd, api):
    self.cmd = cmd
    self.api = api
    self.proc_obj = None

  def run(self):
    print(self.cmd)
    self.proc_obj = subprocess.Popen([self.cmd], shell=True)

  def poll(self):
    assert(self.proc_obj is not None) # should be called from started process
    return self.proc_obj.poll()

  def wait(self):
    assert(self.proc_obj is not None) # should be called from started process
    return self.proc_obj.wait()


class SequentialScheduler:
  def __init__(self, procs):
    self.procs = procs
    self.erroneous = []

  def run(self):
    for proc in self.procs:
      proc.run()
      proc.wait()
      returncode = proc.poll()
      assert(returncode is not None)
      if returncode != 0:
        print(f"CRASH: {proc.api}")


def seed_exist(api, itv_mode, itv_start, itv_end):
  seeds = os.listdir(api_corpus_dir(api))

  for seed in seeds:
    if seed.startswith("CRASH_"):
      continue
    pos = seed.find(itv_mode)
    if pos == -1:
      continue
    digit_start = pos + len(itv_mode)
    i = digit_start
    while i < len(seed):
      if not seed[i].isdigit():
        break
      i += 1
    itv_str = seed[digit_start:i]
    assert(itv_str.isdigit())
    if itv_start <= int(itv_str) and int(itv_str) < itv_end:
      return True

  return False


def init_seed_run_procs(cmd_prefix, apis, itv_mode, itv_start, itv_end):
  procs = []
  for api in apis:
    if seed_exist(api, itv_mode, itv_start, itv_end):
      cmd = f"{cmd_prefix}{api}"
      cmd = f"{cmd} --run_only --ignore_exception --verbose 1"
      cmd = f"{cmd} --run_corpus_from_{itv_mode} {itv_start} --run_corpus_to_{itv_mode} {itv_end}"
      cmd = f"{cmd} --corpus {api_corpus_dir(api)} >> {cov_result_dir()}/{api}.txt 2>&1"
      procs.append(Proc(cmd, api))
  return procs


def main():
  args = parse_arg()

  start_time = time.time()

  procs = init_seed_run_procs(args.cmd_prefix, args.apis, args.itv_mode, args.itv_start, args.itv_end)
  if len(procs) == 0:
    print("No seeds to be executed.")
    exit(0)

  cmd_clean = f"python3 {cov_tool_path()} --clean --target_dir {args.target_dir}"
  print(cmd_clean)
  subprocess.run([cmd_clean], shell=True)

  SequentialScheduler(procs).run()

  cmd_geninfo = f"python3 {cov_tool_path()} --geninfo --out {args.out} --target_dir {args.target_dir}"
  if args.third_party_dir:
    cmd_geninfo = f"{cmd_geninfo} --third_party_dir {args.third_party_dir}"
  print(cmd_geninfo)
  subprocess.run([cmd_geninfo], shell=True)

  end_time = time.time()
  elapsed_time_s = end_time - start_time
  elapsed_time_m = elapsed_time_s / 60
  elapsed_time_h = elapsed_time_m / 60
  elapsed_str = f"\ntime elpased for fuzzing : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
  print(f"\n{elapsed_str}")

if __name__ == "__main__":
  main()
