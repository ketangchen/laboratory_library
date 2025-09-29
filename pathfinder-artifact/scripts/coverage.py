import argparse
import os
import subprocess
from pathlib import Path

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--out",
    type=str,
  )
  parser.add_argument(
    "--show_each",
    action="store_true",
  )
  parser.add_argument(
    "--rm_input",
    action="store_true",
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help=f"clean *.gcda",
  )
  parser.add_argument(
    "--geninfo",
    action="store_true",
  )
  parser.add_argument(
    "--merge",
    type=str,
    nargs='+',
  )
  parser.add_argument(
    "--genhtml",
    type=str,
  )
  parser.add_argument(
    "--lcov_cmd",
    type=str,
    default="lcov",
    help="lcov command. default is `lcov`.",
  )
  parser.add_argument(
    "--genhtml_cmd",
    type=str,
    default="genhtml",
    help="genhtml command. default is `genhtml`.",
  )
  parser.add_argument(
    "--target_dir",
    type=str,
    help="target dir (includes sub-dirs).",
  )
  parser.add_argument(
    "--third_party_dir",
    type=str,
    help="third party dir.",
  )
  parser.add_argument(
    "--include_third_party",
    action="store_false",
    help=f"get coverage of third_parties",
  )
  parser.add_argument(
    "--include_other_libs",
    action="store_false",
    help=f"get coverage of libraries outside of tensorflow",
  )

  return parser.parse_args()


def clean(target_dir):
    subprocess.check_call(
        [
            "find",
            target_dir,
            "-name",
            "*.gcda",
            "-type",
            "f",
            "-delete",
        ]
    )


def geninfo(lcov_cmd,
            target_dir,
            out_file_path,
            third_party_dir,
            include_third_party,
            include_other_libs):
    ignored_pattern = []
    if not include_third_party:
        ignored_pattern += [f"*{third_party_dir}/*"]
    if not include_other_libs:
        ignored_pattern += ["/usr/*", "*anaconda3/*"]

    subprocess.check_call(
        [
            lcov_cmd,
            "--quiet",
            "--branch-coverage",
            "--ignore-errors",
            "mismatch,mismatch",
            "--ignore-errors",
            "gcov,gcov",
            "--ignore-errors",
            "inconsistent,inconsistent",
            "--capture",
            "--directory",
            target_dir,
            "--output-file",
            out_file_path,
        ]
    )

    if len(ignored_pattern) > 0:
        subprocess.check_call(
            [
                lcov_cmd,
                "--quiet",
                "--remove",
                out_file_path,
            ] + ignored_pattern +
            [
                "--branch-coverage",
                "--ignore-errors",
                "mismatch,mismatch",
                "--ignore-errors",
                "unused,unused",
                "--ignore-errors",
                "inconsistent,inconsistent",
                "--output-file",
                out_file_path,
            ]
        )


def merge(lcov_cmd,
          info_file_paths,
          out_file_path,
          show_each,
          rm_input):
    info_file_paths = list(filter(lambda file_path: os.path.isfile(file_path), info_file_paths))

    if len(info_file_paths) == 0:
        print("merge: No valid info files to be merged")
    else:
        if show_each:
            subprocess.check_call(
                [
                    lcov_cmd,
                    "--branch-coverage",
                    "--summary",
                    info_file_paths[0],
                ]
            )

        if len(info_file_paths) == 1:
            info_file_path = info_file_paths[0]
            print(f"merge: Only one input file {info_file_path}. Rename {info_file_path} to {out_file_path}.")
            os.rename(info_file_path, out_file_path)
        else:
            acc = info_file_paths[0]

            for info_file_path in info_file_paths[1:]:
                print(f"merge: Merge `{acc}` and `{info_file_path}` to `{out_file_path}`.")
                subprocess.check_call(
                    [
                        lcov_cmd,
                        "--branch-coverage",
                        "--add-tracefile",
                        acc,
                        "--add-tracefile",
                        info_file_path,
                        "--output-file",
                        out_file_path,
                    ]
                )
                acc = out_file_path

    if rm_input:
        for info_file_path in info_file_paths:
            if os.path.isfile(info_file_path) and info_file_path != out_file_path:
                print(f"merge: Remove `{info_file_path}`")
                os.remove(info_file_path)

def genhtml(genhtml_cmd, info_file_path, out_dir_path):
    if not os.path.isfile(info_file_path):
        print(f"genhtml: Input file {info_file_path} is invalid.")
        exit(0)

    subprocess.check_call(
        [
            genhtml_cmd,
            "--quiet",
            "--branch-coverage",
            info_file_path,
            "--output-directory",
            out_dir_path
        ]
    )


def main():
    args = parse_arg()

    if args.clean:
        clean(args.target_dir)
        exit(0)
    elif args.geninfo:
        geninfo(args.lcov_cmd,
                args.target_dir,
                args.out,
                args.third_party_dir,
                args.include_third_party,
                args.include_other_libs)
    elif args.merge:
        merge(args.lcov_cmd,
              args.merge,
              args.out,
              args.show_each,
              args.rm_input)
    elif args.genhtml:
        genhtml(args.genhtml_cmd,
                args.genhtml,
                args.out)


if __name__ == "__main__":
  main()
