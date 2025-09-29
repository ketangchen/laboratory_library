import os
import pathlib
import shutil
import time
import subprocess
from pathlib import Path
from functools import reduce
from utils import *
from dll_info import *


class FuzzerJob:
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 mem, use_conda):
        self.image_name = image_name
        self.container_name = container_name
        self.container_working_dir = container_working_dir
        self.container_env_vars = container_env_vars
        self.host_fuzz_result_path = host_fuzz_result_path
        self.container_fuzz_result_dir = container_fuzz_result_dir
        self.host_gcov_result_path = host_gcov_result_path
        self.container_cov_result_dir = container_cov_result_dir
        self.mem = mem
        self.use_conda = use_conda
        self.proc = None
        self.skip = False

    def check_dirs(self):
        pass

    def make_dirs(self):
        pass

    def container_env_var_flag(self):
        return reduce(lambda acc, env_var: f"{acc} --env {env_var}", self.container_env_vars, "")

    def container_volume_flag(self):
        volumes = []
        if self.host_fuzz_result_path and self.container_fuzz_result_dir:
            volumes.append(f"{self.host_fuzz_result_path}:{self.container_fuzz_result_dir}")
        if self.host_gcov_result_path and self.container_cov_result_dir:
            volumes.append(f"{self.host_gcov_result_path}:{self.container_cov_result_dir}")
        assert(len(volumes) > 0)
        return reduce(lambda acc, volume: f"{acc} -v {volume}", volumes, "")

    def command(self):
        pass

    def docker_run_cmd(self):
        cmd = f"docker run -itd --cpus 1 -m {self.mem}g "
        cmd = f"{cmd} -w {self.container_working_dir}"
        cmd = f"{cmd} {self.container_env_var_flag()}"
        cmd = f"{cmd} --name {self.container_name}"
        cmd = f"{cmd} {self.container_volume_flag()}"
        cmd = f"{cmd} {self.image_name}"
        cmd = f"{cmd} sh -c \"{self.command()}\""
        return cmd

    def run(self):        
        cmd = self.docker_run_cmd()
        print(cmd)
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

    def wait(self):
        assert(self.proc is not None) # should be called from started process
        return self.proc.wait()

    def container_exists(self):
        cmd = f"docker inspect {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _ = self.proc.communicate()
        return self.proc.returncode == 0

    def is_running(self):
        cmd = f"docker inspect {self.container_name}" + " --format='{{.State.Running}}'"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = self.proc.communicate()
        return out.decode().strip() == "true"

    def exitcode(self):
        cmd = f"docker inspect {self.container_name}" + " --format='{{.State.ExitCode}}'"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = self.proc.communicate()
        out = out.decode().strip()
        if out.isdecimal():
            return int(out)
        else:
            return 1

    def stop(self):
        cmd = f"docker stop {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.proc.wait()

    def rm(self):
        cmd = f"docker rm {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.proc.wait()


class FuzzRunJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 fuzzer_cmd,
                 time_budget,
                 mem, use_conda):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         host_fuzz_result_path,
                         container_fuzz_result_dir,
                         None,
                         None,
                         mem, use_conda)
        self.fuzzer_cmd = fuzzer_cmd
        self.time_budget = time_budget

    def check_dirs(self):
        if os.path.isdir(self.host_fuzz_result_path):
            print(f"Note: Skip `{self.container_name}`, as fuzz result exists in `{self.host_fuzz_result_path}`.")
            self.skip = True

    def make_dirs(self):
        os.makedirs(self.host_fuzz_result_path, exist_ok=True)
        host_corpus_dir = os.path.join(self.host_fuzz_result_path, "corpus")
        os.makedirs(host_corpus_dir, exist_ok=True)

    def loose_timeout(self):
        timeout_margin = 180
        return self.time_budget + timeout_margin

    def timeout_cmd(self):
        return f"timeout {self.loose_timeout()}"

    def command(self):
        cmd = f"{self.timeout_cmd()} {self.fuzzer_cmd}"
        if self.use_conda:
            cmd = f"conda run -n base {cmd}"
        return cmd

    def record_exitcode(self, exitcode):
        assert(os.path.isdir(self.host_fuzz_result_path))
        with open(os.path.join(self.host_fuzz_result_path, "exitcode.txt"), "w") as f:
            f.write(str(exitcode))


class GcovRunJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 covrun_cmd_prefix,
                 itv_mode, itv_start, itv_end, job_id, apis,
                 mem, use_conda):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         host_fuzz_result_path,
                         container_fuzz_result_dir,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, use_conda)
        self.covrun_cmd_prefix = covrun_cmd_prefix
        self.itv_mode = itv_mode
        self.itv_start = itv_start
        self.itv_end = itv_end
        self.job_id = job_id
        self.apis = apis

    def check_dirs(self):
        if not os.path.isdir(self.host_fuzz_result_path):
            print(emphasize(f"Error: Result dir `{self.host_fuzz_result_path}` does not exists."))
            exit(0)
        if os.path.isdir(self.host_gcov_result_path):
            print(emphasize(f"Error: Result dir `{self.host_gcov_result_path}` exists."))
            exit(0)

    def make_dirs(self):
        os.makedirs(self.host_gcov_result_path, exist_ok=True)

    def job_id_str(self):
        return ("0" + str(self.job_id) if self.job_id < 10 else str(self.job_id))

    def out_file_name(self):
        return f"coverage{self.job_id_str()}.info"

    def out_file_rel_path(self):
        subdir = f"{self.itv_mode}{self.itv_start}_{self.itv_end}"
        return os.path.join(subdir, self.out_file_name())

    def command(self):
        apis_str = reduce(lambda a, b: f"{a} {b}", self.apis)

        cmd = self.covrun_cmd_prefix
        cmd = f"{cmd} --itv_mode {self.itv_mode}"
        cmd = f"{cmd} --itv_start {self.itv_start}"
        cmd = f"{cmd} --itv_end {self.itv_end}"
        cmd = f"{cmd} --apis {apis_str}"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/covrun_{self.job_id_str()}.log 2>&1"

        if self.use_conda:
            cmd = f"conda run -n base {cmd}"

        return cmd


class GcovMergeIntraJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 itv_subdir, job_id, intra_cov_jobs,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        assert(len(intra_cov_jobs) >= 2)
        self.itv_subdir = itv_subdir
        self.job_id = job_id
        self.intra_cov_jobs = intra_cov_jobs

    def info_file_paths(self):
        paths = list(map(lambda intra_cov_job: intra_cov_job.out_file_name(), self.intra_cov_jobs))
        paths = list(map(lambda path: os.path.join(self.container_cov_result_dir, path), paths))
        return paths

    def job_id_str(self):
        return ("0" + str(self.job_id) if self.job_id < 10 else str(self.job_id))

    def out_file_name(self):
        return f"merged{self.job_id_str()}.info"

    def out_file_rel_path(self):
        return os.path.join(self.itv_subdir, self.out_file_name())

    def command(self):
        merge_flag = "--merge " + reduce(lambda a, b: f"{a} {b}", self.info_file_paths())

        cmd = f"{container_cov_tool_cmd()} --rm_input {merge_flag}"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/merge_intra_{self.job_id_str()}.log 2>&1"
        return cmd

class GcovMergeInterJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 intra_cov_jobs,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        self.intra_cov_jobs = intra_cov_jobs

    def out_file_name(self):
        return f"merged.info"

    def info_file_paths(self):
        paths = list(map(lambda intra_cov_job: intra_cov_job.out_file_rel_path(), self.intra_cov_jobs))
        paths = list(map(lambda path: os.path.join(self.container_cov_result_dir, path), paths))
        return paths

    def command(self):
        merge_flag = "--merge " + reduce(lambda a, b: f"{a} {b}", self.info_file_paths())

        cmd = f"{container_cov_tool_cmd()} {merge_flag} --show_each"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/merge_inter.log 2>&1"
        return cmd


class GcovGenHtmlJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 inter_cov_job,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        self.inter_cov_job = inter_cov_job

    def info_file_path(self):
        return os.path.join(self.container_cov_result_dir, self.inter_cov_job.out_file_name())

    def command(self):
        genhtml_flag = "--genhtml " + self.info_file_path()
        out_dir_path = os.path.join(self.container_cov_result_dir, f"html")

        cmd = f"{container_cov_tool_cmd()} {genhtml_flag}"
        cmd = f"{cmd} --out {out_dir_path}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/genhtml.log 2>&1"
        return cmd


class JobScheduler:
    def __init__(self,
                 dll_info,
                 ablation,
                 time_budget,
                 repetition_indexes,
                 apis,
                 mem):
        self.dll_info = dll_info
        self.ablation = ablation
        self.time_budget = time_budget
        self.repetition_indexes = repetition_indexes
        self.apis = apis
        self.mem = mem
        
    def set_repeatition_indexes(self):
        repeat_max = 5

        if len(self.repetition_indexes) == 0:
            self.repetition_indexes = range(repeat_max)

    def schedule_worker(self):
        pass

    def schedule(self):
        self.set_repeatition_indexes()

        generations = self.schedule_worker()
        for generation in generations:
            for job in generation:
                job.check_dirs()
        return generations

class FuzzJobScheduler(JobScheduler):
    def __init__(self,
                 dll_info,
                 asan,
                 ablation,
                 time_budget,
                 repetition_indexes,
                 apis,
                 mem):
        super().__init__(dll_info,
                         ablation,
                         time_budget,
                         repetition_indexes,
                         apis,
                         mem)
        self.asan = asan
        if self.asan:
            self.dockerfile_name = self.dll_info.dockerfile_name("asan")
            self.image_name = self.dll_info.image_name("asan")
            self.host_result_base = os.path.join(root_asan_result_path(),
                                                 self.dll_info.host_asan_result_dir(self.ablation, self.time_budget))
            self.container_working_dir = self.dll_info.container_working_dir("asan")
            self.container_env_vars = self.dll_info.container_env_vars("asan")
        else:
            self.dockerfile_name = self.dll_info.dockerfile_name("fuzz")
            self.image_name = self.dll_info.image_name("fuzz")
            self.host_result_base = os.path.join(root_fuzz_result_path(),
                                       self.dll_info.host_fuzz_result_dir(self.ablation, self.time_budget))
            self.container_working_dir = self.dll_info.container_working_dir("fuzz")
            self.container_env_vars = self.dll_info.container_env_vars("fuzz")
        self.use_conda = self.dll_info.use_conda

    def fuzzrun_job(self, repetition_index, api):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        container_name = f"pf-{self.dockerfile_name}{ablation_str}-{self.time_budget}sec-{repetition_index}-{api}"
        host_fuzz_result_path = os.path.join(self.host_result_base, f"{repetition_index}", api)

        fuzzer_cmd = f"{self.dll_info.fuzzer_cmd_prefix(self.ablation)}{api} {self.dll_info.fuzzer_flag(self.asan, self.ablation, self.time_budget)}"
        return FuzzRunJob(self.image_name,
                          container_name,
                          self.container_working_dir,
                          self.container_env_vars,
                          host_fuzz_result_path,
                          self.dll_info.container_fuzz_result_dir,
                          fuzzer_cmd,
                          self.time_budget,
                          self.mem, self.use_conda)

    def schedule_worker(self):
        jobs = []
        for repetition_index in self.repetition_indexes:
            for api in self.apis:
                jobs.append(self.fuzzrun_job(repetition_index, api))
        return [jobs]


class GcovJobScheduler(JobScheduler):
    def __init__(self,
                 dll_info,
                 ablation,
                 time_budget,
                 itv, itv_total, vs,
                 gen_html,
                 repetition_indexes,
                 apis,
                 cpu_capacity, mem):
        super().__init__(dll_info,
                         ablation,
                         time_budget,
                         repetition_indexes,
                         apis,
                         mem)
        assert(itv_total % itv == 0)
        assert(cpu_capacity is not None)

        self.itv_mode = "time"
        self.itv = itv
        self.itv_total = itv_total
        self.vs = vs
        self.gen_html = gen_html
        self.cpu_capacity = cpu_capacity

        self.dockerfile_name = self.dll_info.dockerfile_name("gcov")
        self.image_name = self.dll_info.image_name("gcov")
        self.host_fuzz_result_dir = self.dll_info.host_fuzz_result_dir(self.ablation, self.time_budget)
        self.host_gcov_result_dir = self.dll_info.host_gcov_result_dir(self.ablation, self.time_budget,
                                                                       self.itv, self.itv_total, self.vs)
        self.container_fuzz_result_dir = self.dll_info.container_fuzz_result_dir
        self.container_cov_result_dir = self.dll_info.container_cov_result_dir
        self.container_working_dir = self.dll_info.container_working_dir("gcov")
        self.container_env_vars = self.dll_info.container_env_vars("gcov")
        self.use_conda = self.dll_info.use_conda

    def itv_str(self, itv_start, itv_end):
        return f"{self.itv_mode}{itv_start}_{itv_end}"

    def covrun_job(self, repetition_index, job_id, apis,
                   itv_start, itv_end):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-gcovrun-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}-{self.itv_str(itv_start, itv_end)}-{job_id}"
        host_fuzz_result_path = os.path.join(root_fuzz_result_path(),
                                             self.host_fuzz_result_dir,
                                             f"{repetition_index}")
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}", self.itv_str(itv_start, itv_end))

        runner = os.path.join(container_home_path(), "pathfinder_coverage.py")
        covrun_cmd_prefix = f"python3 -u {runner}"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --cmd_prefix '{self.dll_info.fuzzer_cmd_prefix(self.ablation)}'"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --target_dir {self.dll_info.gcov_target_dir()}"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --third_party_dir {self.dll_info.third_party_dir()}"

        return GcovRunJob(self.image_name,
                          container_name,
                          self.container_working_dir,
                          self.container_env_vars,
                          host_fuzz_result_path,
                          self.container_fuzz_result_dir,
                          host_gcov_result_path,
                          self.container_cov_result_dir,
                          covrun_cmd_prefix,
                          self.itv_mode, itv_start, itv_end, job_id, apis,
                          self.mem, self.use_conda)

    def merge_intra_job(self, repetition_index, job_id,
                        itv_start, itv_end, intra_cov_jobs):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-mergeintra-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}-{self.itv_str(itv_start, itv_end)}-{job_id}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}", self.itv_str(itv_start, itv_end))
        itv_subdir = self.itv_str(itv_start, itv_end)
        return GcovMergeIntraJob(self.image_name,
                                 container_name,
                                 self.container_working_dir,
                                 container_env_vars,
                                 host_gcov_result_path,
                                 self.container_cov_result_dir,
                                 itv_subdir, job_id, intra_cov_jobs,
                                 self.mem)

    def merge_inter_job(self, repetition_index, intra_cov_jobs):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-mergeinter-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}")
        return GcovMergeInterJob(self.image_name,
                                 container_name,
                                 self.container_working_dir,
                                 container_env_vars,
                                 host_gcov_result_path,
                                 self.container_cov_result_dir,
                                 intra_cov_jobs,
                                 self.mem)

    def genhtml_job(self, repetition_index, inter_cov_job):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-genhtml-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}")
        return GcovGenHtmlJob(self.image_name,
                              container_name,
                              self.container_working_dir,
                              container_env_vars,
                              host_gcov_result_path,
                              self.container_cov_result_dir,
                              inter_cov_job,
                              self.mem)

    def num_interval(self):
        return self.itv_total // self.itv

    def jobs_per_itv_optimal(self):
        if (self.cpu_capacity < self.num_interval()):
            return

        n_apis = len(self.apis)
        n_jobs_per_itv = self.cpu_capacity // self.num_interval()
        max_apis_per_job = (n_apis // n_jobs_per_itv) + (0 if (n_apis % n_jobs_per_itv == 0) else 1)
        min_jobs_for_same_max_apis_per_job = (n_apis // max_apis_per_job) + (0 if (n_apis % max_apis_per_job == 0) else 1)
        return min_jobs_for_same_max_apis_per_job

    def __schedule_covrun_jobs_few_cores(self, repetition_index):
        n_core = self.cpu_capacity
        n_itv = self.num_interval()
        assert(n_core < n_itv)
        
        jobs_list = []

        for itv_idx in range(n_itv):
            itv_start = itv_idx * self.itv
            itv_end = itv_start + self.itv
            job_id = 0 # `job_id` is for distinguish jobs in the same interval.
            job = self.covrun_job(repetition_index, job_id, self.apis, itv_start, itv_end)
            jobs_list.append([job])

        return jobs_list

    def schedule_covrun_jobs(self, repetition_index):
        if self.cpu_capacity < self.num_interval():
            return self.__schedule_covrun_jobs_few_cores(repetition_index)

        jobs_list = []

        n_itv = self.itv_total // self.itv
        jobs_per_itv = self.jobs_per_itv_optimal()
        api_list = list(self.apis)

        for itv_idx in range(n_itv):
            api_idx_next = 0
            jobs = []
            for job_id in range(jobs_per_itv):
                n_apis = (len(api_list) // jobs_per_itv) + (1 if job_id <= (len(api_list) % jobs_per_itv - 1) else 0)
                apis = api_list[api_idx_next:api_idx_next + n_apis]
                api_idx_next += n_apis

                itv_start = itv_idx * self.itv
                itv_end = itv_start + self.itv
                job = self.covrun_job(repetition_index, job_id, apis, itv_start, itv_end)
                jobs.append(job)
                job_id += 1
            jobs_list.append(jobs)
        return jobs_list


    def schedule_merge_intra_generations(self, repetition_index, itv_idx, covrun_jobs):
        itv_start = itv_idx * self.itv
        itv_end = itv_start + self.itv

        generations = []

        generation_curr = covrun_jobs
        generations.append(generation_curr)

        job_id = 0
        unpaired = None
        while len(generation_curr) >= 2 or (len(generation_curr) == 1 and unpaired is not None):
            generation_next = []
            for i in range(0, len(generation_curr), 2):
                if i + 1 < len(generation_curr):
                    intra_cov_jobs = [generation_curr[i], generation_curr[i + 1]]
                    job = self.merge_intra_job(repetition_index, job_id, itv_start, itv_end, intra_cov_jobs)
                    job_id += 1
                    generation_next.append(job)
                else:
                    assert(i == len(generation_curr) - 1)

                    if unpaired == None:
                        unpaired = generation_curr[i]
                    else:
                        intra_cov_jobs = [generation_curr[i], unpaired]
                        unpaired = None
                        job = self.merge_intra_job(repetition_index, job_id, itv_start, itv_end, intra_cov_jobs)
                        job_id += 1
                        generation_next.append(job)
            generation_curr = generation_next
            generations.append(generation_curr)
        assert(len(generation_curr) == 1)
        return generations, generation_curr[0]

    def schedule_worker(self):
        generations_total = []
        for repetition_index in self.repetition_indexes:
            generations = []
            itv_intra_cov_jobs = []

            covrun_jobs_list = self.schedule_covrun_jobs(repetition_index)
            for itv_idx, covrun_jobs in enumerate(covrun_jobs_list):
                merge_intra_generations, itv_intra_cov_job = self.schedule_merge_intra_generations(repetition_index, itv_idx, covrun_jobs)
                itv_intra_cov_jobs.append(itv_intra_cov_job)

                for i, merge_intra_generation in enumerate(merge_intra_generations):
                    if len(generations) <= i:
                        generations.append([])
                    generations[i] += merge_intra_generation

            merge_inter_job = self.merge_inter_job(repetition_index, itv_intra_cov_jobs)
            generations.append([merge_inter_job])

            if self.gen_html:
                genhtml_job = self.genhtml_job(repetition_index, merge_inter_job)
                generations.append([genhtml_job])

            generations_total += generations

        return generations_total


class ExpManager:
    def __init__(self,
                 dll_info,
                 mode,
                 asan,
                 ablation,
                 vs,
                 apis,
                 time_budget,
                 rep,
                 itv,
                 itv_total,
                 gen_html,
                 cpu_capacity,
                 mem):
        self.dll_info = dll_info
        self.mode = mode
        self.asan = asan
        self.ablation = ablation
        self.vs = vs
        self.apis = apis
        self.time_budget = time_budget
        self.repetition_indexes = list(range(rep))

        self.itv = itv
        self.itv_total = itv_total
        self.gen_html = gen_html

        self.cpu_capacity = cpu_capacity
        self.mem = mem

        self.check_images()

    def check_images(self):
        available = True

        if self.mode == "fuzz":
            if self.asan:
                available = self.check_image(["base", "asan"]) and available
                if self.dll_info.name == "torch":
                    available = self.check_image(["pov-base"]) and available
            else:
                available = self.check_image(["base", "fuzz"]) and available
        elif self.mode == "gcov":
            available = self.check_image(["base", "gcov"]) and available

        if not available:
            exit(0)

    def check_image(self, deps):
        target_img_name = self.dll_info.image_name(deps[-1])
        cmd = f"docker images -q {target_img_name}"
        proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        image_available = out.decode().strip() != ""

        img_pull_cmd = f"docker pull {target_img_name}"

        base_dockerfile_name = "base"
        base_img_name = f"starlabunist/pathfinder:{base_dockerfile_name}"
        base_dockerfile_path = os.path.join(experiment_home_path(), "docker", f"{base_dockerfile_name}.Dockerfile")
        image_build_cmds = [f"docker build -f {base_dockerfile_path} -t {base_img_name} ."]
        for suffix in deps:
            dockerfile_name = self.dll_info.dockerfile_name(suffix)
            img_name = self.dll_info.image_name(suffix)
            dockerfile_path = os.path.join(experiment_home_path(), "docker", f"{dockerfile_name}.Dockerfile")
            image_build_cmds.append(f"docker build -f {dockerfile_path} -t {img_name} .")

        if not image_available:
            print(emphasize(f"ERROR: Image `{target_img_name}` doesn't exist."))
            print(f"Option 1: Pull the image from dockerhub (recommended).\n")
            print(f"          {img_pull_cmd}\n")
            print(f"Option 2: Build images from Dockerfiles.\n")
            for cmd in image_build_cmds:
                print(f"          {cmd}")
            print("\n")

        return image_available


    def schedule(self):
        self.apis = self.dll_info.set_apis(self.vs, self.apis, self.mode, self.asan)

        if self.mode == "fuzz":
            scheduler = FuzzJobScheduler(self.dll_info,
                                         self.asan,
                                         self.ablation,
                                         self.time_budget,
                                         self.repetition_indexes,
                                         self.apis,
                                         self.mem)
        elif self.mode == "gcov":
            scheduler = GcovJobScheduler(self.dll_info,
                                         self.ablation,
                                         self.time_budget,
                                         self.itv, self.itv_total, self.vs,
                                         self.gen_html,
                                         self.repetition_indexes,
                                         self.apis,
                                         self.cpu_capacity, self.mem)
        return scheduler.schedule()

    def summary_gcov(self):
        print("\n------------------------------------------------")
        print("Summarizing GCOV results...")
        print("------------------------------------------------")

        assert(self.mode == "gcov")
        result_dir = os.path.join(root_gcov_result_path(),
                                  self.dll_info.host_gcov_result_dir(self.ablation, self.time_budget,
                                                                     self.itv, self.itv_total, self.vs))

        for rep_idx in self.repetition_indexes:
            rep_dir = os.path.join(result_dir, str(rep_idx))
            assert(os.path.isdir(rep_dir))

            gcov_result_file = os.path.join(rep_dir, "merge_inter.log")
            if os.path.isfile(gcov_result_file):
                print(f"\nGCOV result file: `{gcov_result_file}`")
            else:
                print(f"\nWARNING: Coverage measurement failed. No GCOV result file `{gcov_result_file}`")
                break

            if self.gen_html:
                html_result_dir = os.path.join(rep_dir, "html")
                if os.path.isdir(html_result_dir):
                    print(f"HTML result dir: `{html_result_dir}`")
                else:
                    print(f"WARNING: Generating HTML failed. No HTML result dir `{html_result_dir}`")

            with open(gcov_result_file, "r") as f:
                print("\nGCOV Summary:")
                print("+------------+-----------------+")
                print("| time (sec) | branch coverage |")
                print("+------------+-----------------+")
                lines = list(filter(lambda line: line.startswith("  branches....: "), f.readlines()))
                t = self.itv
                for line in lines:
                    br_cov_start = line.find("(") + 1
                    br_cov_end = line.find(" of ")
                    br_cov = int(line[br_cov_start:br_cov_end])
                    print(f"|{t:>11} |{br_cov:>16} |")
                    print("+------------+-----------------+")
                    t += self.itv

    def gen_pov(self):
        print("\n------------------------------------------------")
        print("Generating PoVs...")
        print("------------------------------------------------")

        assert(self.asan)
        result_dir = os.path.join(root_asan_result_path(),
                                  self.dll_info.host_asan_result_dir(self.ablation, self.time_budget))

        for rep_idx in self.repetition_indexes:
            rep_dir = os.path.join(result_dir, str(rep_idx))
            assert(os.path.isdir(rep_dir))

            buggy = []
            for api_name in os.listdir(rep_dir):
                if api_name not in self.apis:
                    continue
                api_path = os.path.join(rep_dir, api_name)
                if not os.path.isdir(api_path):
                    continue
                exitcode_file = os.path.join(api_path, "exitcode.txt")
                if not os.path.isfile(exitcode_file):
                    continue
                with open(exitcode_file, "r") as f:
                    exitcode = int(f.readline())
                if exitcode is None or exitcode == 0 or exitcode == 124 or exitcode == 137:
                    continue
                print(f"Exit code {exitcode:>3}: {api_name} ({api_path})")
                buggy.append(api_name)

            temp_dir_name = "_buggy_input"
            temp_dir_path = os.path.join(experiment_home_path(), temp_dir_name)
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            os.makedirs(temp_dir_path, exist_ok=True)

            for api_name in buggy:
                corpus_path = os.path.join(rep_dir, api_name, "corpus")
                if not os.path.isdir(corpus_path):
                    print(f"WARNING: Corpus dir `{corpus_path}` does not exist")
                    continue
                for seed in os.listdir(corpus_path):
                    if seed.startswith("CRASH"):
                        seed_path = os.path.join(corpus_path, seed)
                        dst = os.path.join(temp_dir_path, api_name)
                        os.makedirs(dst, exist_ok=True)
                        shutil.copy(seed_path, dst)
                        break

            if len(os.listdir(temp_dir_path)) > 0:
                prefix = self.dll_info.dockerfile_name("pov")
                dockerfile_path = os.path.join(experiment_home_path(), "docker", f"{prefix}-template.Dockerfile")
                ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
                vs_str = f"-vs_{self.vs}" if self.vs != None else ""
                time_budget_str = f"-{self.time_budget}sec"
                image_name = f"{prefix}{ablation_str}{vs_str}{time_budget_str}-{rep_idx}"
                cmd = f"docker build -f {dockerfile_path} -t {image_name} ."
                print("\n" + cmd + "\n")
                proc = subprocess.Popen([cmd], cwd=experiment_home_path(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                _ = proc.communicate()
                if proc.returncode == 0:
                    print(f"Generated docker image `{image_name}`.")
                    print(f"In the docker image, generated PoV source codes are in `{self.dll_info.pov_source_dir_path()}`.\n")

                    print("How to execute PoVs:")
                    print(f"  # Run docker container.")
                    print(f"  docker run -it --rm {image_name} bash")
                    print(f"  # In the docker container, execute each PoV binaries.")
                    print(f"  {self.dll_info.pov_bin_dir_path()}/<POV_BIN>")
                else:
                    print(emphasize(f"ERROR: Failed to generate docker image `{image_name}`"))
                    exit(0)
            else:
                print(f"\nNo buggy input, does not generate PoV")
                continue
            
            shutil.rmtree(temp_dir_path, ignore_errors=True)


class JobRunner:
    def __init__(self, cpu_capacity, generations, prune_volumes=True):
        self.cpu_capacity = cpu_capacity
        self.generations = generations
        self.prune_volumes = prune_volumes

    def load_jobs(self, generation):
        CPU_PER_JOB = 1

        cpu_jobs = []
        for job in generation:
            assert(self.cpu_capacity >= CPU_PER_JOB)
            if not job.skip:
                cpu_jobs.append(job)

        batch_list = []
        while len(cpu_jobs) > 0:
            cpu_cap = self.cpu_capacity
            batch = []

            cpu_jobs_ = []
            for cpu_job in cpu_jobs:
                if cpu_cap >= CPU_PER_JOB:
                    cpu_cap -= CPU_PER_JOB
                    batch.append(cpu_job)
                else:
                    cpu_jobs_.append(cpu_job)
            cpu_jobs = cpu_jobs_

            batch_list.append(batch)

        return batch_list

    def timeover(self, job, elapsed):
        return hasattr(job, 'time_budget') and elapsed > job.loose_timeout() + 60

    def prune(self):
        cmd = f"docker volume prune --force"
        print(cmd)
        proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _ = proc.communicate()
        if proc.returncode == 0:
            print("pruned local volumes seccesfully")
        else:
            print("ERROR: local volumes prunning failed")
            exit(0)

    def run(self):
        print(f"\nTotal Generations: {len(self.generations)}")
        print(f"CPU Capacity: {self.cpu_capacity}")

        result = {}

        for i, generation in enumerate(self.generations):

            generation_start_time = time.time()

            print("\n================================================")
            print(f"Generation [{i + 1}/{len(self.generations)}]")
            print("================================================\n")

            print(f"Total Jobs:{len(generation):>7}")
            n_skipped = 0
            for job in generation:
                if job.skip:
                    n_skipped += 1
            if n_skipped > 0:
                print(f"Skipped Jobs:{n_skipped:>5}")
                print(f"Ready Jobs:{len(generation) - n_skipped:>7}")

            print(f"CPU Capacity:{self.cpu_capacity:>5}")

            batch_list = self.load_jobs(generation)

            print(f"Batchs:{len(batch_list):>11}")

            for j, batch in enumerate(batch_list):
                print("\n------------------------------------------------")
                print(f"Batch [{j + 1}/{len(batch_list)}] start")
                print("------------------------------------------------")

                batch_start_time = time.time()

                for job in batch:
                    job.make_dirs()
                    job.run()

                all_containers_exist = False
                while not all_containers_exist:
                    time.sleep(1)
                    all_containers_exist = True
                    for job in batch:
                        if not job.container_exists():
                            all_containers_exist = False
                            break

                all_terminated = False
                while not all_terminated:
                    all_terminated = True
                    for job in batch:
                        if job.is_running():
                            if self.timeover(job, time.time() - batch_start_time):
                                job.stop()
                            else:
                                all_terminated = False
                                break
                    print(".", end="")
                    time.sleep(10)
                print()

                for job in batch:
                    exitcode = job.exitcode()

                    if type(job) == FuzzRunJob:
                        job.record_exitcode(exitcode)

                    if exitcode != 0:
                        print(f"ABNORMAL TERMINATION (EXIT CODE {exitcode}): {job.container_name}")
                    job.rm()

                    if exitcode not in result.keys():
                        result[exitcode] = []
                    result[exitcode].append(job)

                if self.prune_volumes:
                    self.prune()

                print(f"\nBatch Elapsed Time : {(time.time() - batch_start_time):.2f} sec")

            print(f"\nGeneration Elapsed Time : {(time.time() - generation_start_time):.2f} sec")

        if 124 in result.keys() or 137 in result.keys():
            print("\nAbnormally Terminated Jobs:")

            if 124 in result.keys():
                print(f"  EXIT CODE 124 (timeout):")
                for job in result[124]:
                    print(f"    {job.container_name}")

            if 137 in result.keys():
                print(f"  EXIT CODE 137 (OOM):")
                for job in result[137]:
                    print(f"    {job.container_name}")

        exist_potential_bugs = False
        for exitcode in result.keys():
            if exitcode != 0 and exitcode != 124 and exitcode != 137:
                exist_potential_bugs = True
                break

        if exist_potential_bugs:
            print("\nPotential Buggy Jobs:")

            for exitcode, jobs in result.items():
                if exitcode != 0 and exitcode != 124 and exitcode != 137:
                    print(f"  EXIT CODE {exitcode}:")
                    for job in jobs:
                        print(f"    {job.container_name}")
