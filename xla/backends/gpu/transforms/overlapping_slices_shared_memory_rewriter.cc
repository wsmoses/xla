#include "xla/backends/gpu/transforms/overlapping_slices_shared_memory_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> OverlappingSlicesSharedMemoryRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion &&
          instr->fusion_kind() == HloInstruction::FusionKind::kLoop) {
        
        // Group dynamic slices inside this fusion by their immediate operand.
        // If an operand is sliced multiple times with different offsets = stencil.
        absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>> slices_by_operand;
        
        for (HloInstruction* fused_instr : instr->fused_instructions()) {
          if (fused_instr->opcode() == HloOpcode::kSlice) {
            slices_by_operand[fused_instr->operand(0)].push_back(fused_instr);
          }
        }

        bool has_overlapping_slices_pattern = false;
        for (const auto& [operand, slices] : slices_by_operand) {
          if (slices.size() > 1) {
            has_overlapping_slices_pattern = true;
            break;
          }
        }

        if (has_overlapping_slices_pattern) {
          // Change the fusion kind to kCustom and set backend config
          instr->set_fusion_kind(HloInstruction::FusionKind::kCustom);
          
          GpuBackendConfig gpu_config;
          if (instr->has_backend_config()) {
            auto old_config = instr->backend_config<GpuBackendConfig>();
            if (old_config.ok()) {
              gpu_config = *old_config;
            }
          }
          FusionBackendConfig& backend_config = *gpu_config.mutable_fusion_backend_config();
          backend_config.set_kind("__overlapping_slices");
          
          TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
