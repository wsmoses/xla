#include "xla/backends/gpu/transforms/overlapping_slices_shared_memory_rewriter.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using OverlappingSlicesSharedMemoryRewriterTest = HloHardwareIndependentTestBase;

TEST_F(OverlappingSlicesSharedMemoryRewriterTest, MatchesOverlappingSlicesPattern) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

%fused_computation (param_0: f32[100]) -> f32[10] {
  %param_0 = f32[100]{0} parameter(0)
  %slice1 = f32[10]{0} slice(%param_0), slice={[0:10]}
  %slice2 = f32[10]{0} slice(%param_0), slice={[1:11]}
  ROOT %add = f32[10]{0} add(%slice1, %slice2)
}

ENTRY %main (param_0: f32[100]) -> f32[10] {
  %param_0 = f32[100]{0} parameter(0)
  ROOT %fusion = f32[10]{0} fusion(%param_0), kind=kLoop, calls=%fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OverlappingSlicesSharedMemoryRewriter pass;
  EXPECT_TRUE(pass.Run(module.get()).value());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config, root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__overlapping_slices");
}

TEST_F(OverlappingSlicesSharedMemoryRewriterTest, IgnoresNonOverlappingSlicesPattern) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

%fused_computation (param_0: f32[100], param_1: f32[100]) -> f32[10] {
  %param_0 = f32[100]{0} parameter(0)
  %param_1 = f32[100]{0} parameter(1)
  %slice1 = f32[10]{0} slice(%param_0), slice={[0:10]}
  %slice2 = f32[10]{0} slice(%param_1), slice={[1:11]}
  ROOT %add = f32[10]{0} add(%slice1, %slice2)
}

ENTRY %main (param_0: f32[100], param_1: f32[100]) -> f32[10] {
  %param_0 = f32[100]{0} parameter(0)
  %param_1 = f32[100]{0} parameter(1)
  ROOT %fusion = f32[10]{0} fusion(%param_0, %param_1), kind=kLoop, calls=%fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OverlappingSlicesSharedMemoryRewriter pass;
  EXPECT_FALSE(pass.Run(module.get()).value());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
