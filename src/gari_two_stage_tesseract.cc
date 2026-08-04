#include "gari_two_stage_tesseract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
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
  const size_t physical_detectors = layout.physical_detector_count;
  const size_t all_detectors = physical_detectors + layout.virtual_detector_count;

  prepared->top_error_to_bottom_detector.reserve(layout.barred_error_count);
  prepared->top_error_detectors.reserve(layout.barred_error_count);
  prepared->bottom_error_detectors.reserve(layout.physical_error_count);

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

    std::vector<size_t> physical_targets;
    std::vector<size_t> virtual_targets;
    std::unordered_set<size_t> seen_detectors;
    std::unordered_set<size_t> seen_observables;
    std::vector<stim::DemTarget> top_targets;
    std::vector<stim::DemTarget> bottom_targets;

    for (const stim::DemTarget& target : instruction.target_data) {
      if (target.is_relative_detector_id()) {
        const size_t detector = target.val();
        if (detector >= all_detectors || !seen_detectors.insert(detector).second) {
          throw std::invalid_argument("Invalid or repeated detector target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        if (detector < physical_detectors) {
          physical_targets.push_back(detector);
          top_targets.push_back(stim::DemTarget::relative_detector_id(detector));
        } else {
          virtual_targets.push_back(detector);
          bottom_targets.push_back(
              stim::DemTarget::relative_detector_id(detector - physical_detectors));
        }
      } else if (target.is_observable_id()) {
        const size_t observable = target.val();
        if (!seen_observables.insert(observable).second) {
          throw std::invalid_argument("Repeated observable target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        bottom_targets.push_back(target);
      } else {
        throw std::invalid_argument("Separator or unsupported target in GARI error " +
                                    std::to_string(error_index) + ".");
      }
    }

    if (error_index < layout.physical_error_count) {
      if (!physical_targets.empty()) {
        throw std::invalid_argument(
            "A physical GARI error violates the bottom [I U; I V] block.");
      }
      if (error_index < layout.virtual_detector_count) {
        if (virtual_targets.size() != 1 ||
            virtual_targets[0] != layout.barred_error_to_virtual_detector[error_index]) {
          throw std::invalid_argument(
              "A physical identity-block error does not target its virtual detector.");
        }
      } else if (virtual_targets.size() != 2) {
        throw std::invalid_argument(
            "A physical Y error must target exactly two virtual detectors.");
      }
      prepared->bottom_dem.append_error_instruction(instruction.arg_data[0], bottom_targets,
                                                     instruction.tag);
      std::vector<size_t> local_targets;
      local_targets.reserve(virtual_targets.size());
      for (size_t detector : virtual_targets) {
        local_targets.push_back(detector - physical_detectors);
      }
      prepared->bottom_error_detectors.push_back(std::move(local_targets));
    } else {
      const size_t barred_error = error_index - layout.physical_error_count;
      const size_t expected_virtual = layout.barred_error_to_virtual_detector[barred_error];
      if (physical_targets.empty() || virtual_targets.size() != 1 ||
          virtual_targets[0] != expected_virtual || !seen_observables.empty()) {
        throw std::invalid_argument(
            "A barred GARI error violates the top block or its debt identity.");
      }
      prepared->top_dem.append_error_instruction(instruction.arg_data[0], top_targets,
                                                 instruction.tag);
      prepared->top_error_to_bottom_detector.push_back(expected_virtual - physical_detectors);
      prepared->top_error_detectors.push_back(std::move(physical_targets));

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
        component_output->dem.append_error_instruction(instruction.arg_data[0], component_targets,
                                                       instruction.tag);
        component_output->error_to_top_error.push_back(barred_error);
        component_output->error_detectors.push_back(std::move(local_detectors));
      }
    }
    ++error_index;
  }

  if (error_index != layout.physical_error_count + layout.barred_error_count ||
      prepared->top_dem.count_errors() != layout.barred_error_count ||
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
          } else {
            bottom_decoder_->decode_to_errors(debt);
          }
        }
      };
      if (config_.collect_bottom_timing) {
        const auto start = std::chrono::steady_clock::now();
        decode_bottom();
        result.bottom_decode_time_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      } else {
        decode_bottom();
      }

      if (config_.bottom_backend == GariBottomBackend::Tesseract) {
        outcome.completed = !bottom_decoder_->low_confidence_flag;
        if (outcome.completed) {
          outcome.errors = bottom_decoder_->predicted_errors_buffer;
          if (!reproduces_syndrome(prepared_model_->bottom_error_detectors, outcome.errors, debt,
                                   layout.virtual_detector_count)) {
            throw std::runtime_error(
                "Completed bottom GARI candidate does not reproduce its debt syndrome.");
          }
          outcome.cost = bottom_decoder_->cost_from_errors(outcome.errors);
          outcome.observables = bottom_decoder_->get_flipped_observables(outcome.errors);
        }
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
