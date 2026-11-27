/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"
#include "debug.h"
#include "nccl_tuner.h"

static ncclTuner_v6_t* ncclTuner_v6;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_init(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn,
                                   ncclNvlDomainInfo_v6_t* nvlDomainInfo, ncclTunerConstants_v6_t* constants,
                                   int64_t /*busId*/, int /*nvmlId*/, int /*rank*/) {
  return ncclTuner_v6->init(ctx, commId, nRanks, nNodes, logfn, nvlDomainInfo, constants);
}

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps,
                                          float** collCostTable, int numAlgo, int numProto, int regBuff,
                                          int* nChannels, const uint64_t* /*collectiveIds*/, size_t /*nCollectiveIds*/,
                                          int /*rank*/) {
  return ncclTuner_v6->getCollInfo(context, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto,
                                    regBuff, nChannels);
}

static ncclResult_t ncclTuner_finalize(void* ctx, int /*rank*/) {
  return ncclTuner_v6->finalize(ctx);
}

ncclTuner_t* getNcclTuner_v6(void* lib) {
  ncclTuner_v6 = (ncclTuner_v6_t*)ncclOsDlsym(lib, "ncclTunerPlugin_v6");
  if (ncclTuner_v6) {
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v6)", ncclTuner_v6->name);
    ncclTuner.name = ncclTuner_v6->name;
    ncclTuner.init = ncclTuner_init;
    ncclTuner.getCollInfo = ncclTuner_getCollInfo;
    ncclTuner.finalize = ncclTuner_finalize;
    ncclTuner.getChunkSize = ncclTuner_v6->getChunkSize;
    return &ncclTuner;
  }
  return NULL;
}
