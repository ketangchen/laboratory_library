import argparse
import os
from pathlib import Path

class PassRateResult:
    def __init__(self, dll, result_dir):
        self.dll = dll
        self.result_dir = result_dir
        self.count()

    def norm_api_name(self, api_name):
        if self.dll == "tf":
            return "tf.raw_ops." + api_name

        if self.dll == "torch":
            api_name = api_name.replace(".", "_")
            if api_name.startswith("at_native_"):
                return api_name[len("at_native_"):]

        return api_name

    def get_api_result(self, api_path):
        stat_csv_file = os.path.join(api_path, "stat.csv")
        if not os.path.isfile(stat_csv_file):
            print(f"No profile, skip `{api_path}`")
            return None, None
        with open(stat_csv_file, "r") as f:
            lines = list(map(lambda line: line.strip(), f.readlines()))
            lines = list(filter(lambda line: len(line) > 0, lines))
            lines_passed = list(filter(lambda line: line.startswith("Number of passed inputs,"), lines))
            lines_failed = list(filter(lambda line: line.startswith("Number of failed inputs,"), lines))
            if len(lines_passed) == 0 or len(lines_failed) == 0 or len(lines_passed) != len(lines_failed):
                return None, None
            num_valid = int(lines_passed[-1].split(",")[1])
            num_invalid = int(lines_failed[-1].split(",")[1])
            return num_valid, num_invalid

    def count(self):
        result = {}

        for rep_idx in os.listdir(self.result_dir):
            rep_dir = os.path.join(self.result_dir, rep_idx)
            if not os.path.isdir(rep_dir):
                continue
            if not rep_idx.isdigit():
                continue
            result[rep_idx] = {}
            for api_name in os.listdir(rep_dir):
                api_path = os.path.join(rep_dir, api_name)
                if not os.path.isdir(api_path):
                    continue
                num_valid, num_invalid = self.get_api_result(api_path)
                if num_valid == None or num_invalid == None:
                    continue
                result[rep_idx][self.norm_api_name(api_name)] = (num_valid, num_invalid)

        agg_result = {}
        for _, rep_result in result.items():
            for api_name, (num_valid, num_invalid) in rep_result.items():
                if api_name not in agg_result.keys():
                    agg_result[api_name] = {"cnt": 1, "n_valid": num_valid, "n_invalid": num_invalid}
                else:
                    agg_result[api_name]["cnt"] += 1
                    agg_result[api_name]["n_valid"] += num_valid
                    agg_result[api_name]["n_invalid"] += num_invalid

        avg_result = {}
        for api_name, api_result in agg_result.items():
            avg_valid = api_result["n_valid"] / api_result["cnt"]
            avg_invalid = api_result["n_invalid"] / api_result["cnt"]
            avg_result[api_name] = avg_valid, avg_invalid 

        self.result = avg_result

    def print_passrate(self):
        assert(self.result is not None)

        n_apis = 0
        total_valid = 0
        total_invalid = 0

        for api_name, (num_valid, num_invalid) in self.result.items():
            n_apis += 1
            total_valid += num_valid
            total_invalid += num_invalid

        print(f"==== {self.dll} ====")
        print(f"Num valid:    {total_valid:10.2f}")
        print(f"Num invalid:  {total_invalid:10.2f}")
        print(f"Valid per API: {total_valid / n_apis:10.2f}")
        print(f"Invalid per API: {total_invalid / n_apis:10.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()

    assert(args.dll in ["torch", "tf"])
    assert(os.path.isdir(args.result_path))

    result = PassRateResult(args.dll, args.result_path)
    result.print_passrate()

if __name__ == "__main__":
    main()
