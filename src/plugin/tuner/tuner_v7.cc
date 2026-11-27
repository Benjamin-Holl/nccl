/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"
#include "debug.h"
#include "nccl_tuner.h"

ncclTuner_t* getNcclTuner_v7(void* lib) {
  ncclTuner_t* tuner = (ncclTuner_t*)ncclOsDlsym(lib, "ncclTunerPlugin_v7");
  if (tuner) INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v7)", tuner->name);
  return tuner;
}
