import os
from utils import *

class TargetAPISet:
    def __init__(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.api_list_dir = os.path.join(file_dir, "api_list")
        assert(os.path.isdir(self.api_list_dir))

    def from_file(self, filename):
        filepath = os.path.join(self.api_list_dir, filename)
        check(os.path.isfile(filepath),
              f"ERROR: Invalid api list file `{filepath}`.")
        self.api_set = set()
        with open(filepath, "r") as f:
            for line in map(lambda line: line.strip(), f.readlines()):
                if line == "" or line.startswith("#"):
                    continue
                self.api_set.add(line)
        return self

    def copy(self, other):
        self.api_set = other.api_set
        return self

    def from_list(self, api_list):
        assert(type(api_list) == list)
        self.api_set = set(api_list)
        return self

    def from_set(self, api_set):
        assert(type(api_set) == set)
        self.api_set = api_set
        return self

    def empty(self):
        return len(self.api_set) == 0

    def difference(self, other):
        return TargetAPISet().from_set(self.api_set - other.api_set)

    def intersection(self, other):
        return TargetAPISet().from_set(self.api_set & other.api_set)

    def union(self, other):
        return TargetAPISet().from_set(self.api_set.union(other.api_set))

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersection(other)

    def __len__(self):
        return len(self.api_set)

    def __contains__(self, item):
        return item in self.api_set

    def __iter__(self):
        return self.api_set.__iter__()

    def __next__(self):
        return self.api_set.__next__()

    def __str__(self):
        return str(self.api_set)


class DLLInfo:
    def __init__(self, name, version, use_conda):
        self.name = name
        self.version = version
        self.use_conda = use_conda

        self.container_fuzz_result_dir = os.path.join(container_home_path(), "experiment_result")
        self.container_cov_result_dir = os.path.join(container_home_path(), "coverage_result")

    def dockerfile_name(self, suffix):
        return f"{self.name}{self.version}-{suffix}"

    def image_name(self, suffix):
        return f"starlabunist/pathfinder:{self.dockerfile_name(suffix)}"

    def host_fuzz_result_dir(self, ablation, time_budget):
        directory = self.dockerfile_name("fuzz")
        directory = f"{directory}-{ablation}"
        directory = f"{directory}-{time_budget}sec"
        return directory

    def host_asan_result_dir(self, ablation, time_budget):
        directory = self.dockerfile_name("asan")
        directory = f"{directory}-{ablation}"
        directory = f"{directory}-{time_budget}sec"
        return directory

    def host_gcov_result_dir(self, ablation, time_budget, itv, itv_total, vs=None):
        directory = self.dockerfile_name("gcov")
        directory = f"{directory}-{ablation}"
        if vs != None:
            directory = f"{directory}-{vs}"
        directory = f"{directory}-{time_budget}sec-itv{itv}_total{itv_total}"
        return directory

    def container_working_dir(self, mode):
        pass

    def container_env_vars(self, mode):
        pass

    def fuzzer_cmd_prefix(self, ablation):
        pass

    def fuzzer_flag(self, asan, ablation, time_budget):
        flag =  f"--max_total_time {time_budget} --min -64 --max 64"
        if ablation == "wo_nbp":
            flag = f"{flag} --wo_nbp"
        if not asan:
            flag = f"{flag} --ignore_exception"
        flag = f"{flag} --output_unique --verbose 0"
        flag = f"{flag} --corpus {self.container_fuzz_result_dir}/corpus"
        flag = f"{flag} --output_stat {self.container_fuzz_result_dir}/stat.csv"
        flag = f"{flag} >> {self.container_fuzz_result_dir}/fuzz_log.txt 2>&1"
        return flag

    def gcov_target_dir(self):
        pass

    def third_party_dir(self):
        # Name of third party dir which should be excluded when measuring coverage
        pass

    def pov_source_dir_path(self):
        pass

    def pov_bin_dir_path(self):
        pass

    def set_apis(self, vs, api_list, mode, asan):
        if vs != None:
            apis = TargetAPISet().from_file(f"{self.name}{self.version}-{vs}.txt")
        else:
            if api_list is None:
                api_list = []
            if type(api_list) == str:
                api_list = [api_list]
            assert(type(api_list) == list)
            apis = TargetAPISet().from_list(api_list)

        return apis


class TorchInfo(DLLInfo):
    def __init__(self, version):
        check(version == "2.2" or version == "1.11",
              f"Error: Invalid version `{version}`. Valid torch versions = [2.2, 1.11].")
        use_conda = True
        super().__init__("torch", version, use_conda)

    def container_working_dir(self, mode):
        return os.path.join(container_home_path(), "pytorch")

    def container_env_vars(self, mode):
        return []

    def fuzzer_cmd_prefix(self, ablation):
        if ablation == "wo_nbp":
            ablation = "default"
        assert(ablation == "default" or ablation == "wo_staged")

        return f"{container_home_path()}/pathfinder-torch-{ablation}/build/bin/pathfinder_fuzz_driver_"

    def gcov_target_dir(self):
        return f"{container_home_path()}/pytorch/build/caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native"

    def third_party_dir(self):
        return "third_party"

    def pov_source_dir_path(self):
        return f"{container_home_path()}/pathfinder-torch/pov"

    def pov_bin_dir_path(self):
        return f"{container_home_path()}/pathfinder-torch/build/bin"

    def set_apis(self, vs, api_list, mode, asan):
        apis = super().set_apis(vs, api_list, mode, asan)

        if self.version == "1.11" and vs == "ivysyn" and not asan:
            apis = TargetAPISet().from_file(f"{self.name}{self.version}-{vs}-rq1.txt")

        if self.version == "2.2":
            all_apis = TargetAPISet().from_file(f"{self.name}{self.version}-all.txt")
            kernels = TargetAPISet().from_file(f"{self.name}{self.version}-acetest.txt")
            all_targets = all_apis.union(kernels)
        elif self.version == "1.11":
            all_targets = TargetAPISet().from_file(f"{self.name}{self.version}-ivysyn.txt")
        invalid = apis - all_targets
        check(len(invalid) == 0,
              f"Error: Invalid APIs `{invalid}`.")

        if len(apis) == 0:
            if self.version == "2.2":
                filename = f"{self.name}{self.version}-all.txt"
            elif self.version == "1.11":
                if asan:
                    filename = f"{self.name}{self.version}-ivysyn.txt"
                else:
                    filename = f"{self.name}{self.version}-ivysyn-rq1.txt"
            print(emphasize(f"Target API is not specified. Fuzz all APIs in `api_list/{filename}`."))
            apis = TargetAPISet().from_file(filename)

        if self.version == "2.2" and mode == "gcov":
            block = TargetAPISet().from_file(f"{self.name}{self.version}-block.txt")
            apis = apis - block

        assert(len(apis) > 0)

        cpp_names = []
        for api in apis:
            if api.startswith("torch."):
                cpp_names.append(api.replace(".", "_"))
            elif api.startswith("at::"):
                cpp_names.append(api.replace("::", "_"))
            else:
                # kernel functions (only when comparing with acetest or ivysyn)
                cpp_names.append("at_native_" + api)
        return cpp_names

class TFInfo(DLLInfo):
    def __init__(self, version):
        check(version == "2.16",
              f"Error: Invalid version `{version}`. Valid tf versions = [2.16].")
        use_conda = False
        super().__init__("tf", version, use_conda)

    def container_working_dir(self, mode):
        return os.path.join(container_home_path(), "tensorflow")

    def container_env_vars(self, mode):
        env_vars = ["TF_CPP_MIN_LOG_LEVEL=3"]
        if mode == "gcov":
            env_vars += [
                f"GCOV_PREFIX={container_tensorflow_home_path()}",
                f"GCOV_PREFIX_STRIP=3",
            ]
        return env_vars

    def fuzzer_cmd_prefix(self, ablation):
        if ablation == "wo_nbp":
            ablation = "default"
        assert(ablation == "default" or ablation == "wo_staged")

        return f"{container_home_path()}/tensorflow/bazel-bin/tensorflow/core/kernels/pathfinder/{ablation}/pathfinder_driver_main "

    def gcov_target_dir(self):
        return f"{container_home_path()}/tensorflow/bazel-out/k8-opt/bin/tensorflow/core/kernels"

    def third_party_dir(self):
        return "external"

    def pov_source_dir_path(self):
        return f"{container_home_path()}/tensorflow/tensorflow/core/kernels/pathfinder/pov"

    def pov_bin_dir_path(self):
        return f"{container_home_path()}/tensorflow/bazel-bin/tensorflow/core/kernels/pathfinder/pov"

    def set_apis(self, vs, api_list, mode, asan):
        apis = super().set_apis(vs, api_list, mode, asan)

        all_targets = TargetAPISet().from_file(f"{self.name}{self.version}-all.txt")
        invalid = apis - all_targets
        check(len(invalid) == 0,
              f"Error: Invalid APIs `{invalid}`.")

        if len(apis) == 0:
            filename = f"{self.name}{self.version}-all.txt"
            print(emphasize(f"Target API is not specified. Fuzz all APIs in `{experiment_home_path()}/api_list/{filename}`."))
            apis = TargetAPISet().from_file(filename)

        if mode == "gcov":
            block = TargetAPISet().from_file(f"{self.name}{self.version}-block.txt")
            apis = apis - block

        assert(len(apis) > 0)

        cpp_names = []
        for api in apis:
            assert(api.startswith("tf.raw_ops."))
            cpp_names.append(api[len("tf.raw_ops."):])
        return cpp_names
