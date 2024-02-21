# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Runs cache simulation.

Example usage:

  python3 -m cache_replacement.policy_learning.cache.main \
    --experiment_base_dir=/tmp \
    --experiment_name=sample_belady_llc \
    --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
    --cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/belady.json \
    --memtrace_file=cache_replacement/policy_learning/cache/traces/sample_trace.csv

  Simulates a cache configured by the cache configs with Belady's as the
  replacement policy on the sample trace.
"""
# pylint: enable=line-too-long

import os
import math
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tqdm
from cache_replacement.policy_learning.cache import cache as cache_mod
from cache_replacement.policy_learning.cache import evict_trace as evict
from cache_replacement.policy_learning.cache import memtrace
from cache_replacement.policy_learning.common import config as cfg
from cache_replacement.policy_learning.common import utils
from cache_replacement.policy_learning.bayesian_changepoint_detection.bayesian_changepoint_detection.hazard_functions import constant_hazard
from cache_replacement.policy_learning.bayesian_changepoint_detection.bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from cache_replacement.policy_learning.bayesian_changepoint_detection.bayesian_changepoint_detection import online_likelihoods as online_ll
import numpy as np
import random

from functools import partial
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "cache_configs",
    [
        "cache_replacement/policy_learning/cache/configs/default.json",  # pylint: disable=line-too-long
        "cache_replacement/policy_learning/cache/configs/eviction_policy/lru.json"  # pylint: disable=line-too-long
    ],
    "List of config paths merged front to back for the cache.")
flags.DEFINE_multi_string(
    "config_bindings", [],
    ("override config with key=value pairs "
     "(e.g., eviction_policy.policy_type=greedy)"))
flags.DEFINE_string(
    "experiment_base_dir", "/tmp/experiments",
    "Base directory to store all experiments in. Should not frequently change.")
flags.DEFINE_string(
    "experiment_name", "unnamed",
    "All data related to this experiment is written to"
    " experiment_base_dir/experiment_name.")
flags.DEFINE_string(
    "memtrace_file",
    "cache_replacement/policy_learning/cache/traces/omnetpp_train.csv",
    "Memory trace file path to use.")
flags.DEFINE_integer(
    "tb_freq", 2000, "Number of cache reads between tensorboard logs.")
flags.DEFINE_integer(
    "warmup_period", int(0), "Number of cache reads before recording stats.")
flags.DEFINE_bool(
    "force_overwrite", True,
    ("If true, overwrites directory at "
     " experiment_base_dir/experiment_name if it exists."))

policy_pool = ['random','bip','lfu','lru','mru','lrfu']
policy_prob = [1/6,1/6,1/6,1/6,1/6,1/6]
weight_pool = [1,1,1,1,1,1]
e = 0.3

def log_scalar(tb_writer, key, value, step):
  summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
  tb_writer.add_summary(summary, step)

#change eviction_policy
def change_policy(C1, policy):
  cache_change = cfg.Config.from_files_and_bindings(
      ["cache_replacement/policy_learning/cache/configs/eviction_policy/" + policy + ".json"], FLAGS.config_bindings)
  
  for i in C1._sets:
    i._eviction_policy = cache_mod.Cache.eviction_change(cache_change.get("eviction_policy"))

#random pick
def random_pick(alist,probs):
  x = random.uniform(0,1)
  cp = 0.0
  for item, item_p in zip(alist,probs):
    cp += item_p
    if x < cp:
      break
  i = -1
  for k in alist:
    i += 1
    if item == k:
      break
  return i

#prob_update
def prob_update_down(num,probs,weights):
  weights[num] = weights[num] * (1 - e)
  tmp = 0
  for s in range(0,len(weights)):
    tmp += weights[s]
  for s in range(0,len(probs)):
    probs[s] = weights[s] / tmp
'''
def prob_update_down(num,probs):
  temp = probs
  k = 1 - probs[num]
  for s in range(0,len(probs)):
    if s != num:
      probs[s] = probs[s] / k
  probs[num] = probs[num] * math.exp(-0.3)
  probs[num] = probs[num] / (probs[num] + k)
  k = 1 - probs[num]
  for s in range(0,len(probs)):
    if s != num:
      probs[s] = probs[s] * k 
'''
def prob_update_up(num,probs):
  temp = probs
  k = 1 - probs[num]
  for s in range(0,len(probs)):
    if s != num:
      probs[s] = probs[s] / k
  probs[num] = probs[num] * math.exp(0.3)
  probs[num] = probs[num] / (probs[num] + k)
  k = 1 - probs[num]
  for s in range(0,len(probs)):
    if s != num:
      probs[s] = probs[s] * k  

def main(_):
  # Set up experiment directory
  exp_dir = os.path.join(FLAGS.experiment_base_dir, FLAGS.experiment_name)
  miss_trace_path = os.path.join(exp_dir, "misses.csv")
  evict_trace_path = os.path.join(exp_dir, "evictions.txt")

  data = []
  data1 = []
  data2 = []
  temp = []
  temp1 = []

  cache_config = cfg.Config.from_files_and_bindings(
      FLAGS.cache_configs, FLAGS.config_bindings)
  tensorboard_dir = os.path.join(exp_dir, "tensorboard",cache_config.get("eviction_policy").get("scorer").get("type"))
  tf.disable_eager_execution()
  tb_writer = tf.summary.FileWriter(tensorboard_dir)
  with open(os.path.join(exp_dir, "cache_config.json"), "w") as f:
    cache_config.to_file(f)

  flags_config = cfg.Config({
      "memtrace_file": FLAGS.memtrace_file,
      "tb_freq": FLAGS.tb_freq,
      "warmup_period": FLAGS.warmup_period,
  })
  with open(os.path.join(exp_dir, "flags.json"), "w") as f:
    flags_config.to_file(f)

  logging.info("Config: %s", str(cache_config))
  logging.info("Flags: %s", str(flags_config))

  cache_line_size = cache_config.get("cache_line_size")
  with memtrace.MemoryTrace(
      FLAGS.memtrace_file, cache_line_size=cache_line_size) as trace:
    with memtrace.MemoryTraceWriter(miss_trace_path) as write_trace:
      with evict.EvictionTrace(evict_trace_path, False) as evict_trace:
        def write_to_eviction_trace(cache_access, eviction_decision):
          evict_trace.write(
              evict.EvictionEntry(cache_access, eviction_decision))

        cache = cache_mod.Cache.from_config(cache_config)
        cache1 = cache_mod.Cache.from_config(cache_config)
        cache2 = cache_mod.Cache.from_config(cache_config)
        change_policy(cache2,policy_pool[0])
        # Warm up cache
        for _ in tqdm.tqdm(range(FLAGS.warmup_period), desc="Warming up cache"):
          pc, address = trace.next()
          hit = cache.read(pc, address, [write_to_eviction_trace])
          
          if not hit:
            write_trace.write(pc, address)

          if trace.done():
            raise ValueError()
        
        # Discard warm-up cache statistics
        cache.hit_rate_statistic.reset()
  
        #initialize
        num_reads = 0
        t = -1
        maxes = np.zeros(300 + 1)
        R = np.zeros((300 + 1, 300 + 1))
        R[0, 0] = 1
        a = online_ll.StudentT(alpha=0.1, beta=0.01, kappa=1,mu=0)
        b = partial(constant_hazard, 250)
        Nw = -1
        flag = False #to compare hit_rate when policy change
        p1 = 0 #current hit_rate 
        p2 = 0 #old hit_rate
        r = 0 #policy
        #one_flag = True #keep the first change point

        with tqdm.tqdm(desc="Simulating cache on MemoryTrace") as pbar:
          while not trace.done():
            pc, address = trace.next()
            hit = cache.read(pc, address, [write_to_eviction_trace])
            hit1 = cache1.read(pc, address, [write_to_eviction_trace])
            hit2 = cache2.read(pc, address, [write_to_eviction_trace])

            if not hit:
              write_trace.write(pc, address)

            num_reads += 1

            if num_reads % FLAGS.tb_freq == 0:
              log_scalar(tb_writer, "cache_hit_rate",
                         cache.hit_rate_statistic.success_rate(), num_reads)
              data.append([cache.hit_rate_statistic.success1_rate()])
              data1.append([cache1.hit_rate_statistic.success1_rate()])
              data2.append([cache2.hit_rate_statistic.success1_rate()])
              temp.append(cache.hit_rate_statistic.success_rate())
              temp1.append(cache1.hit_rate_statistic.success_rate())
              #compare hit_rate
              if flag:
                p1 += cache.hit_rate_statistic.success1_rate()
                p2 += cache1.hit_rate_statistic.success1_rate()
                
              t += 1
             
              #print(t,[cache.hit_rate_statistic.success1_rate()])
              
              #BOCD
              predprobs = a.pdf([cache.hit_rate_statistic.success1_rate()])
              H = b(np.array(range(t + 1)))
              R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)
              R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)
              R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
              a.update_theta([cache.hit_rate_statistic.success1_rate()], t=t)
              maxes[t] = R[:, t].argmax()
              
              #change policy
              Nw += 1
              if (Nw >= 30):   
                #change point
                if maxes[Nw - 1] - maxes[Nw] > 1 and maxes[Nw] < 10:
                    #hit rate decrease
                    if data[Nw - 4] > data[Nw] or data[Nw - 3] > data[Nw] or data[Nw - 2] > data[Nw]:
                      #random replace
                      #cancel the procedure is the random
                      if flag:
                        if p1 <= p2:
                          prob_update_down(r, policy_prob,weight_pool)
                          print("YES!")
                      flag = True
                      p1 = 0
                      p2 = 0
                      r = random_pick(policy_pool, policy_prob)
                      change_policy(cache,policy_pool[r])
                      #one_flag = False #keep the first change point
                      print('1111') 
                      print(policy_pool[r])
                      print(policy_prob)
                '''
                #add change points on 10w and 20w
                if(Nw == 50) or (Nw == 100):
                  if flag:
                    if p1 <= p2:
                      prob_update_down(r, policy_prob,weight_pool)
                      print("YES!")
                  flag = True
                  p1 = 0
                  p2 = 0
                  r = random_pick(policy_pool, policy_prob)
                  change_policy(cache,policy_pool[r])
                  print(policy_pool[r])
                  print(policy_prob)
                '''
              cache.hit_rate_statistic.reset1()
              cache1.hit_rate_statistic.reset1()

            pbar.update(1)
          fp = open("cache_replacement/policy_learning/cache/aaa/test11o",'a+')
          print(temp,file=fp)
          fp.close()
          fp = open("cache_replacement/policy_learning/cache/aaa/test11l",'a+')
          print(temp1,file=fp)
          fp.close()

          fig, ax = plt.subplots(3, figsize=[18,16],sharex=True)
          ax[0].plot(data,label='Current', color = 'b')
          ax[0].plot(data1,label='Old', color = 'r')
          ax[0].plot(data2,label='random', color = 'g')
          ax[1].plot(maxes)
          #ax[1].imshow(R, aspect = 'auto', cmap = 'gray_r', origin = 'lower')
          plt.show()
          log_scalar(tb_writer, "cache_hit_rate",
                     cache.hit_rate_statistic.success_rate(), num_reads)
          print(cache.hit_rate_statistic.success_rate())
          print(cache1.hit_rate_statistic.success_rate())
          print(cache2.hit_rate_statistic.success_rate())

  # Force flush, otherwise last writes will be lost.
  tb_writer.flush()

if __name__ == "__main__":
  app.run(main)
