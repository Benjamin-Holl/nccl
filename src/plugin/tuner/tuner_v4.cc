/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "os.h"
#include "debug.h"
#include "checks.h"
#include "nccl_tuner.h"

static ncclTuner_v4_t* ncclTuner_v4;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps,
                                          float** collCostTable, int numAlgo, int numProto, int regBuff,
                                          int* nChannels, const uint64_t* /*collectiveIds*/, size_t /*nCollectiveIds*/,
                                          int /*rank*/) {
  return ncclTuner_v4->getCollInfo(context, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto,
                                    regBuff, nChannels);
}

static ncclResult_t ncclTuner_finalize(void* ctx, int /*rank*/) {
  return ncclTuner_v4->destroy(ctx);
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                   ncclDebugLogger_t logfn, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                   ncclTunerConstants_t* /*constants*/,
                                   int64_t /*busId*/, int /*nvmlId*/, int /*rank*/) {
  NCCLCHECK(ncclTuner_v4->init(nRanks, nNodes, logfn, context));
  ncclTuner.getCollInfo = ncclTuner_getCollInfo;
  ncclTuner.getChunkSize = NULL;
  ncclTuner.finalize = ncclTuner_finalize;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v4(void* lib) {
  ncclTuner_v4 = (ncclTuner_v4_t*)ncclOsDlsym(lib, "ncclTunerPlugin_v4");
  if (ncclTuner_v4) {
    ncclTuner.name = ncclTuner_v4->name;
    ncclTuner.init = ncclTuner_init;

    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v4)", ncclTuner_v4->name);
    return &ncclTuner;
  }
  return NULL;
}
