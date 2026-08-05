#include "gari_two_stage_tesseract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"

namespace {

struct VectorHash {
  size_t operator()(const std::vector<uint64_t>& values) const {
    size_t hash = values.size();
    for (uint64_t value : values) {
      hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

struct BottomOutcome {
  bool completed = false;
  double cost = std::numeric_limits<double>::infinity();
  std::vector<size_t> errors;
  std::vector<int> observables;
};

size_t checked_end(const GariBlockRange& range, const char* label) {
  if (range.count > std::numeric_limits<size_t>::max() - range.offset) {
    throw std::invalid_argument(std::string("GARI ") + label + " range overflows");
  }
  return range.offset + range.count;
}

void preserve_dem_dimensions(stim::DetectorErrorModel& dem, size_t detector_count,
                             size_t observable_count = 0) {
  if (dem.count_detectors() < detector_count) {
    dem.append_detector_instruction(
        {}, stim::DemTarget::relative_detector_id(detector_count - 1), "");
  }
  if (dem.count_observables() < observable_count) {
    dem.append_logical_observable_instruction(
        stim::DemTarget::observable_id(observable_count - 1), "");
  }
}

void validate_partition(const GariBlockRange& first, const GariBlockRange& second,
                        size_t expected_offset, size_t expected_count, const char* label) {
  const size_t expected_end =
      checked_end({.offset = expected_offset, .count = expected_count}, label);
  if (first.offset != expected_offset || second.offset != checked_end(first, label) ||
      checked_end(second, label) != expected_end) {
    throw std::invalid_argument(std::string("GARI D_X/D_Z ") + label +
                                " do not form a contiguous partition");
  }
}

void validate_top_components(const GariTwoStageLayout& layout) {
  if (!layout.top_components.has_value()) {
    return;
  }
  const auto& components = *layout.top_components;
  validate_partition(components.d_x.detector_rows, components.d_z.detector_rows, 0,
                     layout.physical_detector_count, "detector rows");
  validate_partition(components.d_x.barred_error_columns,
                     components.d_z.barred_error_columns, layout.physical_error_count,
                     layout.barred_error_count, "barred error columns");
  validate_partition(components.d_x.debt_detector_rows,
                     components.d_z.debt_detector_rows, layout.physical_detector_count,
                     layout.virtual_detector_count, "debt detector rows");

  for (const GariTopComponentLayout* component : {&components.d_x, &components.d_z}) {
    if (component->barred_error_columns.count != component->debt_detector_rows.count) {
      throw std::invalid_argument(
          "Each GARI top component requires one debt detector per barred column");
    }
    const size_t first_top_error =
        component->barred_error_columns.offset - layout.physical_error_count;
    for (size_t k = 0; k < component->barred_error_columns.count; ++k) {
      if (layout.barred_error_to_virtual_detector[first_top_error + k] !=
          component->debt_detector_rows.offset + k) {
        throw std::invalid_argument(
            "GARI component debt rows disagree with the barred-error mapping");
      }
    }
  }
}

void validate_layout(const stim::DetectorErrorModel& dem, const GariTwoStageLayout& layout) {
  if (layout.physical_detector_count == 0) {
    throw std::invalid_argument("GARI two-stage layout has no physical detectors.");
  }
  if (layout.virtual_detector_count == 0) {
    throw std::invalid_argument("GARI two-stage layout has no virtual detectors.");
  }
  if (layout.barred_error_count != layout.virtual_detector_count) {
    throw std::invalid_argument(
        "GARI two-stage layout requires one barred error per virtual detector.");
  }
  if (layout.physical_error_count < layout.virtual_detector_count) {
    throw std::invalid_argument(
        "GARI two-stage layout is missing physical identity-block errors.");
  }
  if (layout.barred_error_to_virtual_detector.size() != layout.barred_error_count) {
    throw std::invalid_argument(
        "GARI barred-error mapping size does not match barred_error_count.");
  }

  const size_t detector_count =
      layout.physical_detector_count + layout.virtual_detector_count;
  const size_t error_count = layout.physical_error_count + layout.barred_error_count;
  if (dem.count_detectors() != detector_count) {
    throw std::invalid_argument("GARI detector count does not match the two-stage layout.");
  }
  if (dem.count_errors() != error_count) {
    throw std::invalid_argument("GARI error count does not match the two-stage layout.");
  }

  std::vector<bool> seen_virtual(layout.virtual_detector_count, false);
  for (size_t detector : layout.barred_error_to_virtual_detector) {
    if (detector < layout.physical_detector_count || detector >= detector_count) {
      throw std::invalid_argument(
          "GARI barred-error mapping references a non-virtual detector.");
    }
    size_t local_detector = detector - layout.physical_detector_count;
    if (seen_virtual[local_detector]) {
      throw std::invalid_argument(
          "GARI barred-error mapping is not one-to-one with virtual detectors.");
    }
    seen_virtual[local_detector] = true;
  }
  validate_top_components(layout);
}

struct ValidatedGariError {
  std::vector<size_t> physical_targets;
  std::vector<size_t> virtual_targets;
  std::vector<stim::DemTarget> top_targets;
  std::vector<stim::DemTarget> bottom_targets;
};

template <typename OnError>
void for_each_validated_gari_error(const stim::DetectorErrorModel& flat_dem,
                                   const GariTwoStageLayout& layout, OnError&& on_error) {
  const size_t physical_detectors = layout.physical_detector_count;
  const size_t all_detectors = physical_detectors + layout.virtual_detector_count;
  size_t error_index = 0;

  for (const stim::DemInstruction& instruction : flat_dem.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      if (instruction.type == stim::DemInstructionType::DEM_DETECTOR ||
          instruction.type == stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
        continue;
      }
      throw std::invalid_argument("Unsupported instruction in flattened GARI DEM: " +
                                  instruction.str());
    }
    if (instruction.arg_data.size() != 1 || !std::isfinite(instruction.arg_data[0]) ||
        instruction.arg_data[0] <= 0 || instruction.arg_data[0] >= 1) {
      throw std::invalid_argument(
          "GARI error probabilities must be strictly between zero and one.");
    }

    ValidatedGariError error;
    std::unordered_set<size_t> seen_detectors;
    std::unordered_set<size_t> seen_observables;
    for (const stim::DemTarget& target : instruction.target_data) {
      if (target.is_relative_detector_id()) {
        const size_t detector = target.val();
        if (detector >= all_detectors || !seen_detectors.insert(detector).second) {
          throw std::invalid_argument("Invalid or repeated detector target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        if (detector < physical_detectors) {
          error.physical_targets.push_back(detector);
          error.top_targets.push_back(stim::DemTarget::relative_detector_id(detector));
        } else {
          error.virtual_targets.push_back(detector);
          error.bottom_targets.push_back(
              stim::DemTarget::relative_detector_id(detector - physical_detectors));
        }
      } else if (target.is_observable_id()) {
        const size_t observable = target.val();
        if (!seen_observables.insert(observable).second) {
          throw std::invalid_argument("Repeated observable target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        error.bottom_targets.push_back(target);
      } else {
        throw std::invalid_argument("Separator or unsupported target in GARI error " +
                                    std::to_string(error_index) + ".");
      }
    }

    if (error_index < layout.physical_error_count) {
      if (!error.physical_targets.empty()) {
        throw std::invalid_argument(
            "A physical GARI error violates the bottom [I U; I V] block.");
      }
      if (error_index < layout.virtual_detector_count) {
        if (error.virtual_targets.size() != 1 ||
            error.virtual_targets[0] !=
                layout.barred_error_to_virtual_detector[error_index]) {
          throw std::invalid_argument(
              "A physical identity-block error does not target its virtual detector.");
        }
      } else if (error.virtual_targets.size() != 2) {
        throw std::invalid_argument(
            "A physical Y error must target exactly two virtual detectors.");
      }
    } else {
      const size_t barred_error = error_index - layout.physical_error_count;
      const size_t expected_virtual =
          layout.barred_error_to_virtual_detector[barred_error];
      if (error.physical_targets.empty() || error.virtual_targets.size() != 1 ||
          error.virtual_targets[0] != expected_virtual || !seen_observables.empty()) {
        throw std::invalid_argument(
            "A barred GARI error violates the top block or its debt identity.");
      }
    }
    on_error(error_index, instruction, std::move(error));
    ++error_index;
  }

  if (error_index != layout.physical_error_count + layout.barred_error_count) {
    throw std::invalid_argument("GARI DEM validation did not preserve all configured errors.");
  }
}

bool reproduces_syndrome(const std::vector<std::vector<size_t>>& error_detectors,
                         const std::vector<size_t>& selected_errors,
                         const std::vector<uint64_t>& expected_detections,
                         size_t detector_count) {
  std::vector<bool> actual(detector_count, false);
  std::vector<bool> expected(detector_count, false);
  for (uint64_t detector : expected_detections) {
    if (detector >= detector_count || expected[detector]) {
      return false;
    }
    expected[detector] = true;
  }
  for (size_t error : selected_errors) {
    if (error >= error_detectors.size()) {
      return false;
    }
    for (size_t detector : error_detectors[error]) {
      actual[detector] = !actual[detector];
    }
  }
  return actual == expected;
}

TesseractConfig child_config(const stim::DetectorErrorModel& dem,
                             const GariTwoStageConfig& config) {
  TesseractConfig child;
  child.dem = dem;
  child.beam_climbing = false;
  // The bottom always permits revisits; the top overrides this from its
  // top-specific configuration after the common child settings are built.
  child.no_revisit_dets = false;
  child.verbose = config.verbose;
  child.merge_errors = false;
  child.pqlimit = config.pqlimit;
  child.det_penalty = 0;
  child.create_visualization = false;
  child.sparsify_errors = false;
  return child;
}

std::vector<std::vector<size_t>> build_top_orders(const stim::DetectorErrorModel& dem,
                                                  const GariTwoStageConfig& config) {
  if (!config.top_detector_orders.empty()) {
    const size_t detector_count = dem.count_detectors();
    for (const auto& order : config.top_detector_orders) {
      if (order.size() != detector_count) {
        throw std::invalid_argument("GARI explicit top detector order has the wrong size.");
      }
      std::vector<bool> seen(detector_count);
      for (size_t detector : order) {
        if (detector >= detector_count || seen[detector]) {
          throw std::invalid_argument(
              "GARI explicit top detector order must be a permutation.");
        }
        seen[detector] = true;
      }
    }
    return config.top_detector_orders;
  }
  if (config.top_detector_order_method != DetOrder::DetIndex ||
      config.source_to_top_detector.empty()) {
    return build_det_orders(dem, config.num_top_detector_orders,
                            config.top_detector_order_method,
                            config.top_detector_order_seed);
  }
  const size_t detector_count = dem.count_detectors();
  if (config.source_to_top_detector.size() != detector_count) {
    throw std::invalid_argument("GARI source-to-top detector mapping has the wrong size.");
  }
  std::vector<bool> seen(detector_count);
  for (size_t detector : config.source_to_top_detector) {
    if (detector >= detector_count || seen[detector]) {
      throw std::invalid_argument(
          "GARI source-to-top detector mapping must be a permutation.");
    }
    seen[detector] = true;
  }

  std::mt19937_64 rng(config.top_detector_order_seed);
  std::uniform_int_distribution<int> reverse_order(0, 1);
  std::vector<std::vector<size_t>> orders(config.num_top_detector_orders);
  for (auto& order : orders) {
    order = config.source_to_top_detector;
    if (reverse_order(rng)) {
      std::reverse(order.begin(), order.end());
    }
  }
  return orders;
}

std::vector<std::vector<size_t>> filter_top_orders(
    const std::vector<std::vector<size_t>>& top_orders, const GariBlockRange& detector_rows) {
  std::vector<std::vector<size_t>> result(top_orders.size());
  const size_t end = checked_end(detector_rows, "component detector");
  for (size_t order_index = 0; order_index < top_orders.size(); ++order_index) {
    auto& filtered = result[order_index];
    filtered.reserve(detector_rows.count);
    for (size_t detector : top_orders[order_index]) {
      if (detector >= detector_rows.offset && detector < end) {
        filtered.push_back(detector - detector_rows.offset);
      }
    }
    if (filtered.size() != detector_rows.count) {
      throw std::invalid_argument("GARI top order does not cover a complete D_X/D_Z component");
    }
  }
  return result;
}

TesseractConfig top_child_config(const stim::DetectorErrorModel& dem,
                                 const GariTwoStageConfig& config,
                                 std::vector<std::vector<size_t>> detector_orders) {
  TesseractConfig top = child_config(dem, config);
  top.det_beam = config.max_top_beam;
  top.no_revisit_dets = config.top_no_revisit_dets;
  top.det_penalty = config.top_det_penalty;
  top.sparsify_errors = config.top_sparsify_errors;
  top.sparsify_base_degree = config.top_sparsify_base_degree;
  top.sparsify_max_degree = config.top_sparsify_max_degree;
  top.sparsify_reactivate_limit = config.top_sparsify_reactivate_limit;
  top.det_orders = std::move(detector_orders);
  return top;
}

}  // namespace

const char* gari_prior_policy_name(GariPriorPolicy policy) {
  switch (policy) {
    case GariPriorPolicy::ModeN:
      return "modeN";
    case GariPriorPolicy::ModePR:
      return "modePR";
  }
  throw std::invalid_argument("Unknown GARI prior policy");
}

std::vector<double> reconstruct_gari_original_physical_costs(
    const stim::DetectorErrorModel& gari_dem, const GariTwoStageLayout& layout) {
  const stim::DetectorErrorModel flat_dem = gari_dem.flattened();
  validate_layout(flat_dem, layout);

  const size_t error_count = layout.physical_error_count + layout.barred_error_count;
  std::vector<double> guide_costs(error_count);
  std::vector<std::vector<size_t>> physical_virtual_targets(
      layout.physical_error_count);
  for_each_validated_gari_error(
      flat_dem, layout,
      [&](size_t error_index, const stim::DemInstruction& instruction,
          ValidatedGariError&& error) {
        const double probability = instruction.arg_data[0];
        guide_costs[error_index] = std::log1p(-probability) - std::log(probability);
        if (error_index < layout.physical_error_count) {
          physical_virtual_targets[error_index] = std::move(error.virtual_targets);
        }
      });

  std::vector<double> original_costs(
      guide_costs.begin(), guide_costs.begin() + layout.physical_error_count);
  if (layout.prior_policy == GariPriorPolicy::ModeN) {
    return original_costs;
  }

  double cost_scale = 1;
  for (double cost : guide_costs) {
    cost_scale = std::max(cost_scale, std::abs(cost));
  }
  const double tolerance = 1e-10 * cost_scale;
  for (double& cost : guide_costs) {
    if (!std::isfinite(cost) || cost < -tolerance) {
      throw std::invalid_argument(
          "GARI modePR requires finite nonnegative residual and barred costs.");
    }
    cost = std::max(0.0, cost);
  }

  const size_t no_barred_error = std::numeric_limits<size_t>::max();
  std::vector<size_t> virtual_to_barred(layout.virtual_detector_count,
                                        no_barred_error);
  for (size_t barred = 0; barred < layout.barred_error_count; ++barred) {
    const size_t detector = layout.barred_error_to_virtual_detector[barred];
    const size_t local_detector = detector - layout.physical_detector_count;
    virtual_to_barred[local_detector] = barred;
  }

  for (size_t physical = 0; physical < layout.physical_error_count; ++physical) {
    double cost = guide_costs[physical];
    for (size_t detector : physical_virtual_targets[physical]) {
      const size_t local_detector = detector - layout.physical_detector_count;
      const size_t barred = virtual_to_barred.at(local_detector);
      if (barred == no_barred_error) {
        throw std::invalid_argument(
            "GARI modePR physical error targets an unmapped virtual detector.");
      }
      cost += guide_costs[layout.physical_error_count + barred];
    }
    original_costs[physical] = cost;
  }
  return original_costs;
}

class GariTwoStageTesseractDecoder::BottomMatchingDecoder {
 public:
  struct Result {
    double cost;
    std::vector<int> observables;
  };

  explicit BottomMatchingDecoder(const stim::DetectorErrorModel& dem)
      : mwpm_(pm::detector_error_model_to_mwpm(
            dem, pm::NUM_DISTINCT_WEIGHTS, /*ensure_search_flooder_included=*/false,
            /*enable_correlations=*/false)),
        scratch_(mwpm_.flooder.graph.num_observables) {}

  Result decode(const std::vector<uint64_t>& debt) {
    scratch_.reset();
    pm::decode_detection_events(mwpm_, debt, scratch_.obs_crossed.data(), scratch_.weight,
                                /*edge_correlations=*/false);

    Result result{
        .cost = static_cast<double>(scratch_.weight) /
                mwpm_.flooder.graph.normalising_constant,
    };
    for (size_t observable = 0; observable < scratch_.obs_crossed.size(); ++observable) {
      if (scratch_.obs_crossed[observable]) {
        result.observables.push_back(static_cast<int>(observable));
      }
    }
    return result;
  }

 private:
  pm::Mwpm mwpm_;
  pm::ExtendedMatchingResult scratch_;
};

void validate_gari_monolithic_one_way_model(
    const stim::DetectorErrorModel& gari_dem, const GariTwoStageLayout& layout) {
  (void)reconstruct_gari_original_physical_costs(gari_dem, layout);
}

std::shared_ptr<const GariTwoStagePreparedModel> prepare_gari_two_stage_model(
    const stim::DetectorErrorModel& gari_dem, const GariTwoStageLayout& layout,
    bool prepare_top_components) {
  const stim::DetectorErrorModel flat_dem = gari_dem.flattened();
  validate_layout(flat_dem, layout);
  if (prepare_top_components && !layout.top_components.has_value()) {
    throw std::invalid_argument(
        "--gari-split-top requires a mapping containing "
        "gari_two_stage.top_components; regenerate it with the current gari_dem_utils.py");
  }
  auto prepared = std::make_shared<GariTwoStagePreparedModel>();
  prepared->layout = layout;
  prepared->original_physical_costs =
      reconstruct_gari_original_physical_costs(flat_dem, layout);
  const size_t physical_detectors = layout.physical_detector_count;

  prepared->top_error_to_bottom_detector.reserve(layout.barred_error_count);
  prepared->top_error_detectors.reserve(layout.barred_error_count);
  prepared->bottom_error_detectors.reserve(layout.physical_error_count);

  for_each_validated_gari_error(
      flat_dem, layout,
      [&](size_t error_index, const stim::DemInstruction& instruction,
          ValidatedGariError&& error) {
        if (error_index < layout.physical_error_count) {
          prepared->bottom_dem.append_error_instruction(
              instruction.arg_data[0], error.bottom_targets, instruction.tag);
          std::vector<size_t> local_targets;
          local_targets.reserve(error.virtual_targets.size());
          for (size_t detector : error.virtual_targets) {
            local_targets.push_back(detector - physical_detectors);
          }
          prepared->bottom_error_detectors.push_back(std::move(local_targets));
        } else {
          const size_t barred_error = error_index - layout.physical_error_count;
          const size_t expected_virtual =
              layout.barred_error_to_virtual_detector[barred_error];
          prepared->top_dem.append_error_instruction(
              instruction.arg_data[0], error.top_targets, instruction.tag);
          prepared->top_error_to_bottom_detector.push_back(expected_virtual - physical_detectors);
          prepared->top_error_detectors.push_back(std::move(error.physical_targets));

          if (prepare_top_components && layout.top_components.has_value()) {
            const auto& components = *layout.top_components;
            const GariTopComponentLayout* component = nullptr;
            GariPreparedTopComponent* component_output = nullptr;
            if (error_index < checked_end(components.d_x.barred_error_columns,
                                          "D_X barred error")) {
              component = &components.d_x;
              component_output = &prepared->d_x_top;
            } else {
              component = &components.d_z;
              component_output = &prepared->d_z_top;
            }

            const size_t detector_end =
                checked_end(component->detector_rows, "component detector");
            const size_t debt_end =
                checked_end(component->debt_detector_rows, "component debt detector");
            if (expected_virtual < component->debt_detector_rows.offset ||
                expected_virtual >= debt_end) {
              throw std::invalid_argument(
                  "A barred GARI error targets the other component's debt block");
            }
            std::vector<stim::DemTarget> component_targets;
            std::vector<size_t> local_detectors;
            component_targets.reserve(prepared->top_error_detectors.back().size());
            local_detectors.reserve(prepared->top_error_detectors.back().size());
            for (size_t detector : prepared->top_error_detectors.back()) {
              if (detector < component->detector_rows.offset || detector >= detector_end) {
                throw std::invalid_argument(
                    "A barred GARI error crosses between the D_X and D_Z top blocks");
              }
              const size_t local_detector = detector - component->detector_rows.offset;
              component_targets.push_back(stim::DemTarget::relative_detector_id(local_detector));
              local_detectors.push_back(local_detector);
            }
            component_output->dem.append_error_instruction(
                instruction.arg_data[0], component_targets, instruction.tag);
            component_output->error_to_top_error.push_back(barred_error);
            component_output->error_detectors.push_back(std::move(local_detectors));
          }
        }
      });

  if (prepared->top_dem.count_errors() != layout.barred_error_count ||
      prepared->bottom_dem.count_errors() != layout.physical_error_count) {
    throw std::invalid_argument("GARI DEM split did not preserve all configured errors.");
  }
  preserve_dem_dimensions(prepared->top_dem, layout.physical_detector_count);
  preserve_dem_dimensions(prepared->bottom_dem, layout.virtual_detector_count,
                          gari_dem.count_observables());
  if (prepared->top_dem.count_detectors() != layout.physical_detector_count ||
      prepared->bottom_dem.count_detectors() != layout.virtual_detector_count) {
    throw std::invalid_argument("GARI child detector dimensions do not match the layout.");
  }
  if (prepared->top_dem.count_observables() != 0 ||
      prepared->bottom_dem.count_observables() != gari_dem.count_observables()) {
    throw std::invalid_argument(
        "GARI observables must occur on physical errors in the bottom model only.");
  }
  if (prepare_top_components && layout.top_components.has_value()) {
    const auto& components = *layout.top_components;
    preserve_dem_dimensions(prepared->d_x_top.dem, components.d_x.detector_rows.count);
    preserve_dem_dimensions(prepared->d_z_top.dem, components.d_z.detector_rows.count);
    if (prepared->d_x_top.dem.count_errors() !=
            components.d_x.barred_error_columns.count ||
        prepared->d_z_top.dem.count_errors() !=
            components.d_z.barred_error_columns.count ||
        prepared->d_x_top.dem.count_detectors() != components.d_x.detector_rows.count ||
        prepared->d_z_top.dem.count_detectors() != components.d_z.detector_rows.count) {
      throw std::invalid_argument("GARI D_X/D_Z child dimensions do not match the layout");
    }
  }
  return prepared;
}

GariTwoStageTesseractDecoder::GariTwoStageTesseractDecoder(
    std::shared_ptr<const GariTwoStagePreparedModel> prepared_model,
    GariTwoStageConfig config)
    : prepared_model_(std::move(prepared_model)), config_(std::move(config)) {
  if (!prepared_model_) {
    throw std::invalid_argument("GARI two-stage prepared model must not be null.");
  }
  const auto& layout = prepared_model_->layout;
  if (config_.num_top_detector_orders == 0 && config_.top_detector_orders.empty()) {
    throw std::invalid_argument("GARI top detector-order count must be positive.");
  }
  if (config_.top_candidates_per_trial == 0) {
    throw std::invalid_argument("GARI top candidate count must be positive.");
  }
  if (config_.num_bottom_detector_orders == 0 || config_.num_bottom_detector_orders > 2) {
    throw std::invalid_argument("GARI bottom detector-order count must be 1 or 2.");
  }
  if (layout.prior_policy == GariPriorPolicy::ModePR &&
      config_.bottom_backend == GariBottomBackend::PyMatching) {
    throw std::invalid_argument(
        "The PyMatching GARI bottom backend is unavailable with modePR because "
        "it does not return physical errors for original-cost reranking.");
  }
  if (config_.bottom_backend == GariBottomBackend::PyMatching &&
      (config_.bottom_beam != 2 || config_.num_bottom_detector_orders != 1)) {
    throw std::invalid_argument(
        "The PyMatching GARI bottom backend requires the default bottom beam 2 and "
        "one bottom detector order; these Tesseract-only settings are not used.");
  }
  if (config_.max_top_beam >= INF_DET_BEAM || config_.bottom_beam >= INF_DET_BEAM) {
    throw std::invalid_argument("GARI two-stage beams must be below the infinity sentinel.");
  }
  if (config_.split_top && config_.top_sparsify_errors &&
      config_.top_sparsify_reactivate_limit == -1) {
    // Preserve one joint-top auto limit for both components and for the
    // existing sparsify_reactivate_limit statistic.
    const int error_count = static_cast<int>(std::min(
        prepared_model_->top_dem.count_errors(),
        static_cast<uint64_t>(std::numeric_limits<int>::max())));
    config_.top_sparsify_reactivate_limit =
        std::min(suggest_sparsify_reactivate_limit(
                     prepared_model_->top_dem.count_detectors(),
                     config_.top_sparsify_base_degree),
                 error_count);
  }
  const auto top_orders = build_top_orders(prepared_model_->top_dem, config_);
  if (config_.split_top) {
    if (!layout.top_components.has_value()) {
      throw std::invalid_argument(
          "--gari-split-top requires a mapping containing "
          "gari_two_stage.top_components; regenerate it with the current gari_dem_utils.py");
    }
    const auto& components = *layout.top_components;
    if (prepared_model_->d_x_top.dem.count_errors() !=
            components.d_x.barred_error_columns.count ||
        prepared_model_->d_z_top.dem.count_errors() !=
            components.d_z.barred_error_columns.count) {
      throw std::invalid_argument(
          "GARI D_X/D_Z components were not prepared before worker construction");
    }
    d_x_top_decoder_ = std::make_unique<TesseractDecoder>(top_child_config(
        prepared_model_->d_x_top.dem, config_,
        filter_top_orders(top_orders, components.d_x.detector_rows)));
    d_z_top_decoder_ = std::make_unique<TesseractDecoder>(top_child_config(
        prepared_model_->d_z_top.dem, config_,
        filter_top_orders(top_orders, components.d_z.detector_rows)));
  } else {
    top_decoder_ = std::make_unique<TesseractDecoder>(
        top_child_config(prepared_model_->top_dem, config_, top_orders));
  }

  if (config_.bottom_backend == GariBottomBackend::PyMatching) {
    bottom_matching_decoder_ =
        std::make_unique<BottomMatchingDecoder>(prepared_model_->bottom_dem);
  } else {
    TesseractConfig bottom_config = child_config(prepared_model_->bottom_dem, config_);
    bottom_config.det_beam = config_.bottom_beam;
    bottom_config.det_orders.resize(config_.num_bottom_detector_orders);
    for (auto& order : bottom_config.det_orders) {
      order.resize(layout.virtual_detector_count);
      std::iota(order.begin(), order.end(), 0);
    }
    // Detector-index ordering has only two directions. Keep natural first so
    // equal-cost ties preserve the original one-order result.
    if (bottom_config.det_orders.size() == 2) {
      std::reverse(bottom_config.det_orders[1].begin(), bottom_config.det_orders[1].end());
    }

    bottom_decoder_ = std::make_unique<TesseractDecoder>(std::move(bottom_config));
  }
}

GariTwoStageTesseractDecoder::~GariTwoStageTesseractDecoder() = default;

const std::vector<std::vector<size_t>>&
GariTwoStageTesseractDecoder::bottom_detector_orders() const {
  static const std::vector<std::vector<size_t>> no_orders;
  return bottom_decoder_ ? bottom_decoder_->config.det_orders : no_orders;
}

bool GariTwoStageTesseractDecoder::bottom_no_revisit_dets_enabled() const {
  return bottom_decoder_ && bottom_decoder_->config.no_revisit_dets;
}

const TesseractDecoder& GariTwoStageTesseractDecoder::active_top_decoder() const {
  return config_.split_top ? *d_x_top_decoder_ : *top_decoder_;
}

GariTwoStageDecodeResult GariTwoStageTesseractDecoder::decode(
    const std::vector<uint64_t>& top_detections) {
  const auto& layout = prepared_model_->layout;
  std::vector<bool> seen_detection(layout.physical_detector_count, false);
  for (uint64_t detector : top_detections) {
    if (detector >= layout.physical_detector_count) {
      throw std::invalid_argument("Top syndrome references a non-physical detector.");
    }
    if (seen_detection[detector]) {
      throw std::invalid_argument("Top syndrome contains a repeated detector.");
    }
    seen_detection[detector] = true;
  }

  std::vector<uint64_t> d_x_detections;
  std::vector<uint64_t> d_z_detections;
  if (config_.split_top) {
    const auto& components = *layout.top_components;
    const size_t d_x_end = checked_end(components.d_x.detector_rows, "D_X detector");
    for (uint64_t detector : top_detections) {
      if (detector < d_x_end) {
        d_x_detections.push_back(detector - components.d_x.detector_rows.offset);
      } else {
        d_z_detections.push_back(detector - components.d_z.detector_rows.offset);
      }
    }
  }

  GariTwoStageDecodeResult result;
  std::unordered_map<std::vector<uint64_t>, BottomOutcome, VectorHash> bottom_cache;
  auto consider_candidate = [&](const std::vector<size_t>& top_errors, bool validate_top) {
    if (validate_top &&
        !reproduces_syndrome(prepared_model_->top_error_detectors, top_errors, top_detections,
                             layout.physical_detector_count)) {
      throw std::runtime_error("Completed top GARI candidate does not reproduce its syndrome.");
    }

    std::vector<bool> debt_bits(layout.virtual_detector_count, false);
    for (size_t error : top_errors) {
      if (error >= prepared_model_->top_error_to_bottom_detector.size()) {
        throw std::runtime_error("Top decoder returned an invalid barred-error index.");
      }
      size_t detector = prepared_model_->top_error_to_bottom_detector[error];
      debt_bits[detector] = !debt_bits[detector];
    }
    std::vector<uint64_t> debt;
    for (size_t detector = 0; detector < debt_bits.size(); ++detector) {
      if (debt_bits[detector]) {
        debt.push_back(detector);
      }
    }

    BottomOutcome outcome;
    auto cached = bottom_cache.find(debt);
    if (cached != bottom_cache.end()) {
      ++result.bottom_cache_hits;
      outcome = cached->second;
    } else {
      auto consider_tesseract_bottom = [&]() {
        if (bottom_decoder_->low_confidence_flag) {
          return;
        }
        std::vector<size_t> errors = bottom_decoder_->predicted_errors_buffer;
        if (!reproduces_syndrome(prepared_model_->bottom_error_detectors, errors, debt,
                                 layout.virtual_detector_count)) {
          throw std::runtime_error(
              "Completed bottom GARI candidate does not reproduce its debt syndrome.");
        }
        double cost = 0;
        if (layout.prior_policy == GariPriorPolicy::ModePR) {
          for (size_t error : errors) {
            cost += prepared_model_->original_physical_costs.at(error);
          }
        } else {
          cost = bottom_decoder_->cost_from_errors(errors);
        }
        if (!outcome.completed || cost < outcome.cost) {
          outcome.completed = true;
          outcome.cost = cost;
          outcome.errors = std::move(errors);
          outcome.observables =
              bottom_decoder_->get_flipped_observables(outcome.errors);
        }
      };
      auto decode_bottom = [&]() {
        if (config_.bottom_backend == GariBottomBackend::PyMatching) {
          auto matching_result = bottom_matching_decoder_->decode(debt);
          outcome.completed = true;
          outcome.cost = matching_result.cost;
          outcome.observables = std::move(matching_result.observables);
        } else {
          // Preserve the original direct path when bottom diversity is disabled.
          if (config_.num_bottom_detector_orders == 1) {
            bottom_decoder_->decode_to_errors(debt, /*detector_order=*/0,
                                              config_.bottom_beam);
            consider_tesseract_bottom();
          } else {
            for (size_t detector_order = 0;
                 detector_order < config_.num_bottom_detector_orders;
                 ++detector_order) {
              bottom_decoder_->decode_to_errors(
                  debt, detector_order, config_.bottom_beam);
              consider_tesseract_bottom();
            }
          }
        }
      };
      if (config_.collect_bottom_timing) {
        const auto start = std::chrono::steady_clock::now();
        decode_bottom();
        result.bottom_decode_time_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start)
                .count();
      } else {
        decode_bottom();
      }

      bottom_cache.emplace(debt, outcome);
    }

    if (outcome.completed && outcome.cost < result.physical_cost) {
      result.completed = true;
      result.physical_cost = outcome.cost;
      result.observables = std::move(outcome.observables);
      result.top_errors = top_errors;
      result.physical_errors = std::move(outcome.errors);
    }
  };
  auto combine_top_errors = [&](const std::vector<size_t>& d_x_errors,
                                const std::vector<size_t>& d_z_errors) {
    std::vector<size_t> top_errors;
    top_errors.reserve(d_x_errors.size() + d_z_errors.size());
    for (size_t error : d_x_errors) {
      top_errors.push_back(prepared_model_->d_x_top.error_to_top_error.at(error));
    }
    for (size_t error : d_z_errors) {
      top_errors.push_back(prepared_model_->d_z_top.error_to_top_error.at(error));
    }
    std::sort(top_errors.begin(), top_errors.end());
    return top_errors;
  };
  const size_t top_order_count = active_top_decoder().config.det_orders.size();

  const size_t top_trial_count = config_.top_beam_climbing
                                     ? std::max(config_.max_top_beam + 1, top_order_count)
                                     : top_order_count;
  for (size_t trial = 0; trial < top_trial_count; ++trial) {
    const size_t top_beam = config_.top_beam_climbing
                                ? trial % (config_.max_top_beam + 1)
                                : config_.max_top_beam;
    const size_t top_detector_order = trial % top_order_count;

    if (config_.split_top) {
      if (config_.top_candidates_per_trial == 1) {
        d_x_top_decoder_->decode_to_errors(d_x_detections, top_detector_order, top_beam);
        d_z_top_decoder_->decode_to_errors(d_z_detections, top_detector_order, top_beam);
        if (d_x_top_decoder_->low_confidence_flag || d_z_top_decoder_->low_confidence_flag) {
          continue;
        }
        consider_candidate(combine_top_errors(d_x_top_decoder_->predicted_errors_buffer,
                                              d_z_top_decoder_->predicted_errors_buffer),
                           true);
        continue;
      }

      auto d_x_candidates = d_x_top_decoder_->decode_to_error_candidates(
          d_x_detections, top_detector_order, top_beam, config_.top_candidates_per_trial);
      auto d_z_candidates = d_z_top_decoder_->decode_to_error_candidates(
          d_z_detections, top_detector_order, top_beam, config_.top_candidates_per_trial);
      // The blocks are disjoint, so validating each component once validates
      // every pair without repeating the full H_top check K^2 times.
      for (const auto& candidate : d_x_candidates) {
        if (!reproduces_syndrome(prepared_model_->d_x_top.error_detectors, candidate,
                                 d_x_detections,
                                 prepared_model_->d_x_top.dem.count_detectors())) {
          throw std::runtime_error("Completed D_X candidate does not reproduce its syndrome");
        }
      }
      for (const auto& candidate : d_z_candidates) {
        if (!reproduces_syndrome(prepared_model_->d_z_top.error_detectors, candidate,
                                 d_z_detections,
                                 prepared_model_->d_z_top.dem.count_detectors())) {
          throw std::runtime_error("Completed D_Z candidate does not reproduce its syndrome");
        }
      }
      for (const auto& d_x_errors : d_x_candidates) {
        for (const auto& d_z_errors : d_z_candidates) {
          consider_candidate(combine_top_errors(d_x_errors, d_z_errors), false);
        }
      }
    } else {
      if (config_.top_candidates_per_trial == 1) {
        top_decoder_->decode_to_errors(top_detections, top_detector_order, top_beam);
        if (!top_decoder_->low_confidence_flag) {
          consider_candidate(top_decoder_->predicted_errors_buffer, true);
        }
      } else {
        auto top_candidates = top_decoder_->decode_to_error_candidates(
            top_detections, top_detector_order, top_beam, config_.top_candidates_per_trial);
        for (const auto& top_errors : top_candidates) {
          consider_candidate(top_errors, true);
        }
      }
    }
  }
  result.unique_debts = bottom_cache.size();
  return result;
}

namespace {

struct CompactBitsetHash {
  size_t operator()(const boost::dynamic_bitset<>& bits) const {
    return boost::hash_value(bits);
  }
};

struct CompactPhaseWorkspace {
  std::vector<DetectorCostTuple> initial_cost_tuples;
  std::vector<DetectorCostTuple> detector_cost_tuples;
  std::vector<DetectorCostTuple> next_detector_cost_tuples;
  std::vector<double> detector_cost_cache;
  boost::dynamic_bitset<> detectors;
  boost::dynamic_bitset<> next_detectors;

  CompactPhaseWorkspace(size_t num_errors, size_t num_detectors)
      : initial_cost_tuples(num_errors),
        detector_cost_tuples(num_errors),
        next_detector_cost_tuples(num_errors),
        detector_cost_cache(num_detectors),
        detectors(num_detectors),
        next_detectors(num_detectors) {}
};

struct CompactOneWayTrialResult {
  bool completed = false;
  double physical_cost = std::numeric_limits<double>::infinity();
  std::vector<size_t> top_errors;
  std::vector<size_t> physical_errors;
};

struct CompactBottomOutcome {
  bool completed = false;
  double cost = std::numeric_limits<double>::infinity();
  std::vector<size_t> errors;
};

}  // namespace

class GariMonolithicOneWayTesseractDecoder::Impl {
 public:
  Impl(std::shared_ptr<const GariTwoStagePreparedModel> prepared_model,
       GariMonolithicOneWayDecoderConfig config)
      : prepared_model_(std::move(prepared_model)), config_(std::move(config)) {
    if (!prepared_model_) {
      throw std::invalid_argument(
          "GARI one-way prepared model must not be null.");
    }
    if (config_.num_detector_orders == 0 && config_.detector_orders.empty()) {
      throw std::invalid_argument(
          "GARI one-way detector-order count must be positive.");
    }
    if (!std::isfinite(config_.continuation_factor) ||
        config_.continuation_factor < 0) {
      throw std::invalid_argument(
          "GARI one-way continuation factor must be finite and nonnegative.");
    }
    if (config_.beam_climbing && config_.max_beam >= INF_DET_BEAM) {
      throw std::invalid_argument(
          "GARI one-way beam climbing requires a finite beam.");
    }
    debt_bits_.resize(prepared_model_->layout.virtual_detector_count);

    GariTwoStageConfig top_settings;
    top_settings.max_top_beam = config_.max_beam;
    top_settings.num_top_detector_orders = config_.num_detector_orders;
    top_settings.top_detector_order_method = config_.detector_order_method;
    top_settings.top_detector_order_seed = config_.detector_order_seed;
    top_settings.top_no_revisit_dets = config_.no_revisit_dets;
    top_settings.pqlimit = config_.pqlimit;
    top_settings.top_det_penalty = config_.det_penalty;
    top_settings.top_sparsify_errors = config_.sparsify_errors;
    top_settings.top_sparsify_base_degree = config_.sparsify_base_degree;
    top_settings.top_sparsify_max_degree = config_.sparsify_max_degree;
    top_settings.top_sparsify_reactivate_limit =
        config_.sparsify_reactivate_limit;
    top_settings.verbose = config_.verbose;
    top_settings.top_detector_orders = std::move(config_.detector_orders);
    top_settings.source_to_top_detector =
        std::move(config_.source_to_top_detector);
    auto top_orders = build_top_orders(prepared_model_->top_dem, top_settings);
    top_decoder_ = std::make_unique<TesseractDecoder>(top_child_config(
        prepared_model_->top_dem, top_settings, std::move(top_orders)));

    TesseractConfig bottom_config =
        child_config(prepared_model_->bottom_dem, top_settings);
    bottom_config.det_beam = config_.bottom_beam;
    bottom_config.no_revisit_dets = config_.no_revisit_dets;
    bottom_config.merge_errors = config_.bottom_merge_errors;
    bottom_config.det_penalty = config_.det_penalty;
    bottom_config.det_orders.resize(1);
    bottom_config.det_orders[0].resize(
        prepared_model_->layout.virtual_detector_count);
    std::iota(bottom_config.det_orders[0].begin(),
              bottom_config.det_orders[0].end(), 0);
    bottom_decoder_ =
        std::make_unique<TesseractDecoder>(std::move(bottom_config));

    const double unset_cost = std::numeric_limits<double>::quiet_NaN();
    bottom_original_costs_by_error_.assign(bottom_decoder_->num_errors,
                                           unset_cost);
    for (size_t original_error = 0;
         original_error < prepared_model_->original_physical_costs.size();
         ++original_error) {
      const size_t retained_error =
          bottom_decoder_->dem_error_to_error.at(original_error);
      if (retained_error == std::numeric_limits<size_t>::max()) continue;
      double& retained_cost =
          bottom_original_costs_by_error_[retained_error];
      const double original_cost =
          prepared_model_->original_physical_costs[original_error];
      retained_cost = std::isnan(retained_cost)
                          ? original_cost
                          : common::merge_weights(retained_cost, original_cost);
    }

    top_workspace_ = std::make_unique<CompactPhaseWorkspace>(
        top_decoder_->num_errors, top_decoder_->num_detectors);
    bottom_workspace_ = std::make_unique<CompactPhaseWorkspace>(
        bottom_decoder_->num_errors, bottom_decoder_->num_detectors);
    if (config_.pqlimit != std::numeric_limits<size_t>::max()) {
      top_decoder_->error_chain_arena.reserve(config_.pqlimit);
    }
  }

  GariMonolithicOneWayDecodeResult decode(
      const std::vector<uint64_t>& top_detections) {
    stats_ = {};
    bottom_cache_.clear();
    unique_bottom_debts_explored_.clear();
    physical_incumbent_ = std::numeric_limits<double>::infinity();
    GariMonolithicOneWayDecodeResult result;
    if (top_detections.empty()) {
      result.completed = true;
      result.physical_cost = 0;
      result.stats = stats_;
      return result;
    }

    const auto& layout = prepared_model_->layout;
    std::vector<bool> seen_detection(layout.physical_detector_count, false);
    for (uint64_t detector : top_detections) {
      if (detector >= layout.physical_detector_count) {
        throw std::invalid_argument(
            "GARI one-way syndrome references a non-real detector.");
      }
      if (seen_detection[detector]) {
        throw std::invalid_argument(
            "GARI one-way syndrome contains a repeated detector.");
      }
      seen_detection[detector] = true;
    }

    const auto& active_top_d2e = prepare_top_d2e(top_detections);
    boost::dynamic_bitset<> top_state(top_decoder_->num_detectors);
    for (uint64_t detector : top_detections) top_state[detector] = 1;
    const double top_initial_cost = initial_phase_cost(
        *top_decoder_, top_state, active_top_d2e, *top_workspace_);
    if (top_initial_cost == INF) {
      result.stats = stats_;
      return result;
    }
    const size_t order_count = top_decoder_->config.det_orders.size();
    const size_t trial_count =
        config_.beam_climbing
            ? std::max(config_.max_beam + 1, order_count)
            : order_count;
    for (size_t trial = 0; trial < trial_count; ++trial) {
      const size_t beam = config_.beam_climbing
                              ? trial % (config_.max_beam + 1)
                              : config_.max_beam;
      const size_t order = trial % order_count;
      CompactOneWayTrialResult trial_result = run_trial(
          top_state, top_initial_cost, active_top_d2e, order, beam);
      if (trial_result.completed &&
          trial_result.physical_cost < result.physical_cost) {
        result.completed = true;
        result.physical_cost = trial_result.physical_cost;
        result.top_errors = std::move(trial_result.top_errors);
        result.physical_errors = std::move(trial_result.physical_errors);
      }
    }
    if (result.completed) {
      result.observables =
          bottom_decoder_->get_flipped_observables(result.physical_errors);
    }
    result.stats = stats_;
    return result;
  }

  const TesseractDecoder& top_decoder() const {
    return *top_decoder_;
  }

  const TesseractDecoder& bottom_decoder() const {
    return *bottom_decoder_;
  }

 private:
  struct OptionalErrorCandidate {
    int error_index;
    int overlap;
    int degree;
    double likelihood_cost;
  };

  static size_t add_saturating(size_t a, size_t b) {
    const size_t limit = std::numeric_limits<size_t>::max();
    return b > limit - a ? limit : a + b;
  }

  const std::vector<std::vector<int>>& prepare_top_d2e(
      const std::vector<uint64_t>& detections) {
    if (!top_decoder_->config.sparsify_errors) {
      return top_decoder_->d2e;
    }
    if (sparse_d2e_valid_ && sparse_d2e_detections_ == detections &&
        sparse_d2e_reactivate_limit_ ==
            top_decoder_->config.sparsify_reactivate_limit) {
      return top_decoder_->sparse_d2e;
    }

    std::vector<uint8_t> shot_dets(top_decoder_->num_detectors, 0);
    for (uint64_t detector : detections) {
      shot_dets[detector] = 1;
    }
    std::fill(top_decoder_->sparse_error_active.begin(),
              top_decoder_->sparse_error_active.end(), 0);
    for (int error : top_decoder_->sparsify_mandatory_errors) {
      top_decoder_->sparse_error_active[error] = 1;
    }

    std::vector<OptionalErrorCandidate> candidates;
    candidates.reserve(top_decoder_->sparsify_optional_errors.size());
    for (int error : top_decoder_->sparsify_optional_errors) {
      int overlap = 0;
      for (int detector : top_decoder_->errors[error].symptom.detectors) {
        overlap += shot_dets[detector];
      }
      if (overlap > 0) {
        candidates.push_back(
            {error, overlap,
             static_cast<int>(
                 top_decoder_->errors[error].symptom.detectors.size()),
             top_decoder_->errors[error].likelihood_cost});
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const OptionalErrorCandidate& a,
                 const OptionalErrorCandidate& b) {
                if (a.overlap != b.overlap) return a.overlap > b.overlap;
                if (a.degree != b.degree) return a.degree < b.degree;
                if (a.likelihood_cost != b.likelihood_cost) {
                  return a.likelihood_cost < b.likelihood_cost;
                }
                return a.error_index < b.error_index;
              });

    const size_t limit = std::min(
        static_cast<size_t>(top_decoder_->config.sparsify_reactivate_limit),
        candidates.size());
    for (size_t k = 0; k < limit; ++k) {
      top_decoder_->sparse_error_active[candidates[k].error_index] = 1;
    }
    for (size_t detector = 0; detector < top_decoder_->num_detectors;
         ++detector) {
      auto& sparse_errors = top_decoder_->sparse_d2e[detector];
      sparse_errors.clear();
      for (int error : top_decoder_->d2e[detector]) {
        if (top_decoder_->sparse_error_active[error]) {
          sparse_errors.push_back(error);
        }
      }
    }
    sparse_d2e_detections_ = detections;
    sparse_d2e_reactivate_limit_ =
        top_decoder_->config.sparsify_reactivate_limit;
    sparse_d2e_valid_ = true;
    return top_decoder_->sparse_d2e;
  }

  double initial_phase_cost(
      TesseractDecoder& decoder, const boost::dynamic_bitset<>& state,
      const std::vector<std::vector<int>>& active_d2e,
      CompactPhaseWorkspace& workspace) {
    std::fill(workspace.initial_cost_tuples.begin(),
              workspace.initial_cost_tuples.end(), DetectorCostTuple{});
    for (size_t detector = state.find_first();
         detector != boost::dynamic_bitset<>::npos;
         detector = state.find_next(detector)) {
      for (int error : active_d2e[detector]) {
        ++workspace.initial_cost_tuples[error].detectors_count;
      }
    }
    double cost = 0;
    for (size_t detector = state.find_first();
         detector != boost::dynamic_bitset<>::npos;
         detector = state.find_next(detector)) {
      cost += decoder.get_detcost(detector, workspace.initial_cost_tuples,
                                  active_d2e, decoder.edets);
    }
    return cost;
  }

  static std::vector<size_t> collect_chain_errors(
      const TesseractDecoder& decoder, int64_t chain_index) {
    std::vector<size_t> result;
    while (chain_index != -1) {
      const size_t error =
          decoder.error_chain_arena[chain_index].error_index;
      result.push_back(decoder.error_to_dem_error.at(error));
      chain_index = decoder.error_chain_arena[chain_index].parent_idx;
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  CompactOneWayTrialResult run_trial(
      const boost::dynamic_bitset<>& top_state, double top_initial_cost,
      const std::vector<std::vector<int>>& active_top_d2e,
      size_t top_detector_order, size_t top_beam) {
    CompactOneWayTrialResult best;
    top_decoder_->error_chain_arena.clear();
    size_t num_pq_pushed = 0;
    bool pqlimit_hit = false;
    bool top_completion_seen = false;
    size_t continuation_window = 0;
    size_t continuation_deadline = 0;

    using CompletionHandler =
        std::function<bool(const Node&, size_t)>;
    using PhaseRunner = std::function<std::optional<Node>(
        TesseractDecoder&, CompactPhaseWorkspace&,
        const boost::dynamic_bitset<>&, double, size_t,
        const std::vector<std::vector<int>>&, size_t, bool,
        const CompletionHandler&)>;
    PhaseRunner run_phase;

    run_phase = [&](TesseractDecoder& decoder,
                    CompactPhaseWorkspace& workspace,
                    const boost::dynamic_bitset<>& initial_state,
                    double initial_cost, size_t phase_beam,
                    const std::vector<std::vector<int>>& active_d2e,
                    size_t detector_order, bool is_top,
                    const CompletionHandler& completion_handler)
        -> std::optional<Node> {
      if (initial_cost == INF) return std::nullopt;

      std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
      std::unordered_map<
          size_t,
          std::unordered_set<boost::dynamic_bitset<>, CompactBitsetHash>>
          visited_detectors;
      const size_t initial_num_dets = initial_state.count();
      size_t min_num_dets = initial_num_dets;
      size_t max_num_dets = std::min(
          initial_state.size(), add_saturating(initial_num_dets, phase_beam));
      pq.push({initial_cost, initial_num_dets, 0, -1});
      ++num_pq_pushed;
      if (num_pq_pushed > config_.pqlimit) {
        pqlimit_hit = true;
        return std::nullopt;
      }

      std::optional<Node> last_completion;
      size_t phase_queue_pops = 0;
      while (!pq.empty()) {
        if (pqlimit_hit) return last_completion;
        if (is_top && top_completion_seen &&
            config_.continuation_factor > 0 &&
            phase_queue_pops >= continuation_deadline) {
          return last_completion;
        }

        const Node node = pq.top();
        pq.pop();
        ++phase_queue_pops;
        if (config_.collect_stats) {
          if (is_top) {
            ++stats_.top_queue_pops;
            if (top_completion_seen) {
              ++stats_.continuation_top_queue_pops;
            }
          } else {
            ++stats_.bottom_queue_pops;
          }
        }
        if (node.num_dets > max_num_dets) continue;

        workspace.detectors = initial_state;
        std::fill(workspace.detector_cost_tuples.begin(),
                  workspace.detector_cost_tuples.end(), DetectorCostTuple{});
        decoder.flip_detectors_and_block_errors(
            detector_order, node.error_chain_idx, workspace.detectors,
            workspace.detector_cost_tuples, active_d2e, decoder.edets);

        if (node.num_dets == 0) {
          last_completion = node;
          if (is_top && completion_handler(node, phase_queue_pops)) {
            min_num_dets = 0;
            max_num_dets = std::min(max_num_dets, phase_beam);
            continue;
          }
          return node;
        }

        if (decoder.config.no_revisit_dets &&
            !visited_detectors[node.num_dets]
                 .insert(workspace.detectors)
                 .second) {
          continue;
        }
        if (config_.verbose) {
          std::cout.precision(13);
          std::cout << (is_top ? "GARI compact top" : "GARI compact bottom")
                    << ": len(pq) = " << pq.size()
                    << " num_pq_pushed = " << num_pq_pushed
                    << " num_dets = " << node.num_dets
                    << " max_num_dets = " << max_num_dets
                    << " cost = " << node.cost << std::endl;
        }

        if (node.num_dets < min_num_dets) {
          min_num_dets = node.num_dets;
          const size_t new_max_num_dets =
              std::min(initial_state.size(),
                       add_saturating(min_num_dets, phase_beam));
          if (decoder.config.no_revisit_dets) {
            for (size_t count = new_max_num_dets + 1;
                 count <= max_num_dets; ++count) {
              visited_detectors[count].clear();
            }
          }
          max_num_dets = std::min(max_num_dets, new_max_num_dets);
        }

        for (size_t detector = workspace.detectors.find_first();
             detector != boost::dynamic_bitset<>::npos;
             detector = workspace.detectors.find_next(detector)) {
          for (int error : active_d2e[detector]) {
            ++workspace.detector_cost_tuples[error].detectors_count;
          }
        }
        workspace.next_detector_cost_tuples =
            workspace.detector_cost_tuples;
        workspace.next_detectors = workspace.detectors;

        size_t min_detector = std::numeric_limits<size_t>::max();
        for (size_t detector : decoder.config.det_orders[detector_order]) {
          if (workspace.detectors[detector]) {
            min_detector = detector;
            break;
          }
        }
        if (min_detector == std::numeric_limits<size_t>::max()) {
          throw std::runtime_error(
              "A nonempty compact GARI phase has no active detector pivot.");
        }

        size_t previous_error = std::numeric_limits<size_t>::max();
        std::fill(workspace.detector_cost_cache.begin(),
                  workspace.detector_cost_cache.end(), -1);
        for (int error : active_d2e[min_detector]) {
          if (workspace.detector_cost_tuples[error].error_blocked) continue;

          if (previous_error != std::numeric_limits<size_t>::max()) {
            for (int detector : decoder.edets[previous_error]) {
              workspace.next_detectors[detector] =
                  !workspace.next_detectors[detector];
              const int fired = workspace.detectors[detector] ? 1 : -1;
              for (int other_error : active_d2e[detector]) {
                workspace.next_detector_cost_tuples[other_error]
                    .detectors_count += fired;
              }
            }
          }
          previous_error = error;
          workspace.next_detector_cost_tuples[error].error_blocked = 1;

          double next_cost =
              node.cost + decoder.errors[error].likelihood_cost;
          size_t next_num_dets = node.num_dets;
          for (int detector : decoder.edets[error]) {
            workspace.next_detectors[detector] =
                !workspace.next_detectors[detector];
            const int fired = workspace.next_detectors[detector] ? 1 : -1;
            next_num_dets += fired;
            for (int other_error : active_d2e[detector]) {
              workspace.next_detector_cost_tuples[other_error]
                  .detectors_count += fired;
            }
          }
          if (next_num_dets > max_num_dets) continue;
          if (decoder.config.no_revisit_dets &&
              visited_detectors[next_num_dets].contains(
                  workspace.next_detectors)) {
            continue;
          }

          for (int detector : decoder.edets[error]) {
            if (workspace.detectors[detector]) {
              if (workspace.detector_cost_cache[detector] == -1) {
                workspace.detector_cost_cache[detector] = decoder.get_detcost(
                    detector, workspace.detector_cost_tuples, active_d2e,
                    decoder.edets);
              }
              next_cost -= workspace.detector_cost_cache[detector];
            } else {
              next_cost += decoder.get_detcost(
                  detector, workspace.next_detector_cost_tuples, active_d2e,
                  decoder.edets);
            }
          }
          for (int detector : decoder.eneighbors[error]) {
            if (!workspace.detectors[detector] ||
                !workspace.next_detectors[detector]) {
              continue;
            }
            if (workspace.detector_cost_cache[detector] == -1) {
              workspace.detector_cost_cache[detector] = decoder.get_detcost(
                  detector, workspace.detector_cost_tuples, active_d2e,
                  decoder.edets);
            }
            next_cost -= workspace.detector_cost_cache[detector];
            next_cost += decoder.get_detcost(
                detector, workspace.next_detector_cost_tuples, active_d2e,
                decoder.edets);
          }
          if (next_cost == INF) continue;

          decoder.error_chain_arena.push_back(
              {static_cast<size_t>(error), min_detector,
               node.error_chain_idx});
          const int64_t next_chain_index =
              decoder.error_chain_arena.size() - 1;
          pq.push({next_cost, next_num_dets, node.depth + 1,
                   next_chain_index});
          ++num_pq_pushed;
          if (config_.collect_stats && !is_top) {
            ++stats_.bottom_children_generated;
            if (next_num_dets == node.num_dets) {
              ++stats_.bottom_nonprogress_children_generated;
            }
          }
          if (num_pq_pushed > config_.pqlimit) {
            pqlimit_hit = true;
            return last_completion;
          }
        }
      }
      return last_completion;
    };

    CompletionHandler handle_top_completion =
        [&](const Node& top_solution, size_t top_queue_pops) {
          const bool first_completion = !top_completion_seen;
          if (config_.collect_stats) ++stats_.top_completions_seen;
          debt_bits_.reset();
          int64_t top_chain_index = top_solution.error_chain_idx;
          while (top_chain_index != -1) {
            const size_t retained_error =
                top_decoder_->error_chain_arena[top_chain_index].error_index;
            const size_t error =
                top_decoder_->error_to_dem_error.at(retained_error);
            const size_t detector =
                prepared_model_->top_error_to_bottom_detector.at(error);
            debt_bits_.flip(detector);
            top_chain_index =
                top_decoder_->error_chain_arena[top_chain_index].parent_idx;
          }
          std::vector<uint64_t> debt;
          for (size_t detector = debt_bits_.find_first();
               detector != boost::dynamic_bitset<>::npos;
               detector = debt_bits_.find_next(detector)) {
            debt.push_back(detector);
          }

          CompactBottomOutcome outcome;
          if (debt.empty()) {
            outcome.completed = true;
            outcome.cost = 0;
          } else {
            if (config_.collect_stats) ++stats_.bottom_contexts_generated;
            auto cached = bottom_cache_.find(debt);
            if (cached != bottom_cache_.end()) {
              outcome = cached->second;
              if (config_.collect_stats) ++stats_.bottom_cache_hits;
            } else {
              const double bottom_initial_cost = initial_phase_cost(
                  *bottom_decoder_, debt_bits_, bottom_decoder_->d2e,
                  *bottom_workspace_);
              if (bottom_initial_cost != INF) {
                bottom_decoder_->error_chain_arena.clear();
                const size_t bottom_pops_before = stats_.bottom_queue_pops;
                const double committed_top_cost =
                    prepared_model_->layout.prior_policy ==
                            GariPriorPolicy::ModePR
                        ? top_solution.cost
                        : 0;
                const CompletionHandler stop_at_first_bottom;
                std::optional<Node> bottom_solution = run_phase(
                    *bottom_decoder_, *bottom_workspace_, debt_bits_,
                    committed_top_cost + bottom_initial_cost,
                    config_.bottom_beam, bottom_decoder_->d2e,
                    /*detector_order=*/0, /*is_top=*/false,
                    stop_at_first_bottom);
                if (config_.collect_stats &&
                    stats_.bottom_queue_pops > bottom_pops_before) {
                  ++stats_.bottom_contexts_explored;
                  unique_bottom_debts_explored_.insert(debt);
                  stats_.unique_bottom_debts_explored =
                      unique_bottom_debts_explored_.size();
                }
                if (bottom_solution.has_value()) {
                  outcome.completed = true;
                  outcome.errors = collect_chain_errors(
                      *bottom_decoder_,
                      bottom_solution->error_chain_idx);
                  outcome.cost = 0;
                  for (size_t error : outcome.errors) {
                    const size_t retained_error =
                        bottom_decoder_->dem_error_to_error.at(error);
                    const double cost =
                        bottom_original_costs_by_error_.at(retained_error);
                    if (!std::isfinite(cost)) {
                      throw std::runtime_error(
                          "A compact GARI bottom error has no original cost.");
                    }
                    outcome.cost += cost;
                  }
                }
              }
              if (outcome.completed) {
                bottom_cache_.emplace(debt, outcome);
              }
            }
          }

          if (outcome.completed && outcome.cost < best.physical_cost) {
            best.completed = true;
            best.physical_cost = outcome.cost;
            best.top_errors = collect_chain_errors(
                *top_decoder_, top_solution.error_chain_idx);
            best.physical_errors = outcome.errors;
          }
          const bool improved_shot =
              outcome.completed && outcome.cost < physical_incumbent_;
          if (improved_shot) physical_incumbent_ = outcome.cost;

          if (first_completion) {
            top_completion_seen = true;
            if (config_.continuation_factor <= 0 || pqlimit_hit) return false;
            const long double scaled = std::ceil(
                static_cast<long double>(config_.continuation_factor) *
                top_queue_pops);
            continuation_window =
                scaled >= static_cast<long double>(
                              std::numeric_limits<size_t>::max())
                    ? std::numeric_limits<size_t>::max()
                    : std::max<size_t>(1, static_cast<size_t>(scaled));
            continuation_deadline =
                add_saturating(top_queue_pops, continuation_window);
            return true;
          }
          if (improved_shot) {
            continuation_deadline =
                add_saturating(top_queue_pops, continuation_window);
            if (config_.collect_stats) {
              ++stats_.continuation_physical_improvements;
            }
          }
          return !pqlimit_hit;
        };

    std::optional<Node> top_solution = run_phase(
        *top_decoder_, *top_workspace_, top_state, top_initial_cost, top_beam,
        active_top_d2e, top_detector_order, /*is_top=*/true,
        handle_top_completion);
    if (config_.collect_stats && pqlimit_hit) {
      ++stats_.pqlimit_truncated_trials;
    }
    if ((!top_solution.has_value() || !best.completed) && config_.verbose &&
        !pqlimit_hit) {
      std::cout << (top_solution.has_value()
                        ? "GARI compact bottom search failed to converge."
                        : "GARI compact top search failed to converge.")
                << std::endl;
    }
    return best;
  }

  std::shared_ptr<const GariTwoStagePreparedModel> prepared_model_;
  GariMonolithicOneWayDecoderConfig config_;
  std::unique_ptr<TesseractDecoder> top_decoder_;
  std::unique_ptr<TesseractDecoder> bottom_decoder_;
  std::unique_ptr<CompactPhaseWorkspace> top_workspace_;
  std::unique_ptr<CompactPhaseWorkspace> bottom_workspace_;
  std::vector<double> bottom_original_costs_by_error_;
  boost::dynamic_bitset<> debt_bits_;
  std::unordered_map<std::vector<uint64_t>, CompactBottomOutcome, VectorHash>
      bottom_cache_;
  std::unordered_set<std::vector<uint64_t>, VectorHash>
      unique_bottom_debts_explored_;
  GariMonolithicOneWayStats stats_;
  double physical_incumbent_ = std::numeric_limits<double>::infinity();
  bool sparse_d2e_valid_ = false;
  int sparse_d2e_reactivate_limit_ = -1;
  std::vector<uint64_t> sparse_d2e_detections_;
};

GariMonolithicOneWayTesseractDecoder::
    GariMonolithicOneWayTesseractDecoder(
        std::shared_ptr<const GariTwoStagePreparedModel> prepared_model,
        GariMonolithicOneWayDecoderConfig config)
    : prepared_model_(std::move(prepared_model)) {
  impl_ = std::make_unique<Impl>(prepared_model_, std::move(config));
}

GariMonolithicOneWayTesseractDecoder::~GariMonolithicOneWayTesseractDecoder() =
    default;

GariMonolithicOneWayDecodeResult
GariMonolithicOneWayTesseractDecoder::decode(
    const std::vector<uint64_t>& top_detections) {
  return impl_->decode(top_detections);
}

const stim::DetectorErrorModel&
GariMonolithicOneWayTesseractDecoder::compact_top_dem() const {
  return prepared_model_->top_dem;
}

const stim::DetectorErrorModel&
GariMonolithicOneWayTesseractDecoder::compact_bottom_dem() const {
  return prepared_model_->bottom_dem;
}

const std::vector<std::vector<size_t>>&
GariMonolithicOneWayTesseractDecoder::top_detector_orders() const {
  return impl_->top_decoder().config.det_orders;
}

size_t GariMonolithicOneWayTesseractDecoder::compact_top_detector_count()
    const {
  return impl_->top_decoder().num_detectors;
}

size_t GariMonolithicOneWayTesseractDecoder::compact_top_error_count() const {
  return impl_->top_decoder().num_errors;
}

size_t GariMonolithicOneWayTesseractDecoder::compact_bottom_detector_count()
    const {
  return impl_->bottom_decoder().num_detectors;
}

size_t GariMonolithicOneWayTesseractDecoder::compact_bottom_error_count() const {
  return impl_->bottom_decoder().num_errors;
}

bool GariMonolithicOneWayTesseractDecoder::top_no_revisit_dets_enabled()
    const {
  return impl_->top_decoder().config.no_revisit_dets;
}

bool GariMonolithicOneWayTesseractDecoder::bottom_no_revisit_dets_enabled()
    const {
  return impl_->bottom_decoder().config.no_revisit_dets;
}

bool GariMonolithicOneWayTesseractDecoder::top_merge_errors_enabled() const {
  return impl_->top_decoder().config.merge_errors;
}

bool GariMonolithicOneWayTesseractDecoder::bottom_merge_errors_enabled() const {
  return impl_->bottom_decoder().config.merge_errors;
}

int GariMonolithicOneWayTesseractDecoder::top_sparsify_reactivate_limit()
    const {
  return impl_->top_decoder().config.sparsify_reactivate_limit;
}
