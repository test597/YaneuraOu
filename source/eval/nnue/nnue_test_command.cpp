// NNUE評価関数に関するUSI拡張コマンド

#include "../../config.h"

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../extra/all.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace YaneuraOu {
namespace Eval::NNUE {

namespace {

// 主に差分計算に関するRawFeaturesのテスト
void TestFeatures(Position& pos) {
  const std::uint64_t num_games = 1000;
  StateInfo si;
  pos.set_hirate(&si);
  const int MAX_PLY = 256; // 256手までテスト

  StateInfo state[MAX_PLY]; // StateInfoを最大手数分だけ
  int ply; // 初期局面からの手数

  PRNG prng(20171128);

  std::uint64_t num_moves = 0;
  std::vector<std::uint64_t> num_updates(kRefreshTriggers.size() + 1);
  std::vector<std::uint64_t> num_resets(kRefreshTriggers.size());
  constexpr IndexType kUnknown = -1;
  std::vector<IndexType> trigger_map(RawFeatures::kDimensions, kUnknown);
  auto make_index_sets = [&](const Position& pos) {
    std::vector<std::vector<std::set<IndexType>>> index_sets(
        kRefreshTriggers.size(), std::vector<std::set<IndexType>>(2));
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : COLOR) {
        for (const auto index : active_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT(index_sets[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          index_sets[i][perspective].insert(index);
          trigger_map[index] = i;
        }
      }
    }
    return index_sets;
  };
  auto update_index_sets = [&](const Position& pos, auto* index_sets) {
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : COLOR) {
        if (reset[perspective]) {
          (*index_sets)[i][perspective].clear();
          ++num_resets[i];
        } else {
          for (const auto index : removed_indices[perspective]) {
            ASSERT(index < RawFeatures::kDimensions);
            ASSERT((*index_sets)[i][perspective].count(index) == 1);
            ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
            (*index_sets)[i][perspective].erase(index);
            ++num_updates.back();
            ++num_updates[i];
            trigger_map[index] = i;
          }
        }
        for (const auto index : added_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions);
          ASSERT((*index_sets)[i][perspective].count(index) == 0);
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i);
          (*index_sets)[i][perspective].insert(index);
          ++num_updates.back();
          ++num_updates[i];
          trigger_map[index] = i;
        }
      }
    }
  };

  std::cout << "feature set: " << RawFeatures::GetName()
            << "[" << RawFeatures::kDimensions << "]" << std::endl;
  std::cout << "start testing with random games";

  for (std::uint64_t i = 0; i < num_games; ++i) {
    auto index_sets = make_index_sets(pos);
    for (ply = 0; ply < MAX_PLY; ++ply) {
      MoveList<LEGAL_ALL> mg(pos); // 全合法手の生成

      // 合法な指し手がなかった == 詰み
      if (mg.size() == 0)
        break;

      // 生成された指し手のなかからランダムに選び、その指し手で局面を進める。
      Move m = mg.begin()[prng.rand(mg.size())];
      pos.do_move(m, state[ply]);

      ++num_moves;
      update_index_sets(pos, &index_sets);
      ASSERT(index_sets == make_index_sets(pos));
    }

    pos.set_hirate(&si);

    // 100回に1回ごとに'.'を出力(進んでいることがわかるように)
    if ((i % 100) == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves, "
            << num_updates.back() << " updates, "
            << (1.0 * num_updates.back() / num_moves)
            << " updates per move" << std::endl;
  std::size_t num_observed_indices = 0;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    const auto count = std::count(trigger_map.begin(), trigger_map.end(), i);
    num_observed_indices += count;
    std::cout << "TriggerEvent(" << static_cast<int>(kRefreshTriggers[i])
              << "): " << count << " features ("
              << (100.0 * count / RawFeatures::kDimensions) << "%), "
              << num_updates[i] << " updates ("
              << (1.0 * num_updates[i] / num_moves) << " per move), "
              << num_resets[i] << " resets ("
              << (100.0 * num_resets[i] / num_moves) << "%)"
              << std::endl;
  }
  std::cout << "observed " << num_observed_indices << " ("
            << (100.0 * num_observed_indices / RawFeatures::kDimensions)
            << "% of " << RawFeatures::kDimensions
            << ") features" << std::endl;
}

// 評価関数の構造を表す文字列を出力する
void PrintInfo(std::istream& stream) {
  std::cout << "network architecture: " << GetArchitectureString() << std::endl;

  while (true) {
    std::string file_name;
    stream >> file_name;
    if (file_name.empty()) break;

    std::uint32_t hash_value;
    std::string architecture;
    const Tools::Result result = [&]() {
      std::ifstream file_stream(file_name, std::ios::binary);
      if (!file_stream) return Tools::Result(Tools::ResultCode::FileReadError);
	  return ReadHeader(file_stream, &hash_value, &architecture);
    }();

    std::cout << file_name << ": ";
    if (result.is_ok()) {
      if (hash_value == kHashValue) {
        std::cout << "matches with this binary";
        if (architecture != GetArchitectureString()) {
          std::cout << ", but architecture string differs: " << architecture;
        }
        std::cout << std::endl;
      } else {
        std::cout << architecture << std::endl;
      }
    } else {
      std::cout << "failed to read header" << std::endl;
    }
  }
}


// ----------------------------------
//   "test nn cache" : ネットワークパラメーターがL1/L2/L3キャッシュに載るかを調べる
// ----------------------------------

// 1つのキャッシュの情報。size == 0 は検出できなかったことを意味する。
struct CacheLevelInfo {
  std::size_t size = 0;         // 容量[bytes]
  std::size_t shared_cpus = 0;  // このキャッシュを共有している論理CPUの数

  bool operator<(const CacheLevelInfo& rhs) const {
    return std::tie(size, shared_cpus) < std::tie(rhs.size, rhs.shared_cpus);
  }
};

// L1データキャッシュ, L2, L3 の順
constexpr int kCacheLevels = 3;
using CacheLevels = std::array<CacheLevelInfo, kCacheLevels>;
const char* const kCacheLevelNames[kCacheLevels] = {"L1d", "L2", "L3"};

// キャッシュ構成が同じ論理CPUをひとまとめにしたもの
struct CacheGroup {
  CacheLevels levels;
  std::vector<std::size_t> cpus;
};

// 論理CPUごとのキャッシュ構成をOSのAPIで取得する。
// 取得できなかった場合は空のmapが返る。
std::map<std::size_t, CacheLevels> DetectCachePerCpu() {
  std::map<std::size_t, CacheLevels> per_cpu;

#if defined(__linux__) && !defined(__ANDROID__)

  // "0-3,8,10-11" のようなCPUリストを展開する。
  auto parse_cpu_list = [](const std::string& s) {
    std::vector<std::size_t> cpus;
    std::istringstream is(s);
    std::string range;
    while (std::getline(is, range, ',')) {
      if (range.empty() || !std::isdigit(static_cast<unsigned char>(range[0])))
        continue;
      const auto hyphen = range.find('-');
      const std::size_t first = std::stoul(range.substr(0, hyphen));
      const std::size_t last  = hyphen == std::string::npos ? first : std::stoul(range.substr(hyphen + 1));
      for (std::size_t c = first; c <= last; ++c)
        cpus.push_back(c);
    }
    return cpus;
  };

  // "48K", "2048K", "32M" のようなサイズ表記をbytesに変換する。
  auto parse_size = [](const std::string& s) -> std::size_t {
    std::size_t i = 0, value = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
      value = value * 10 + std::size_t(s[i++] - '0');
    if (i < s.size()) {
      switch (std::toupper(static_cast<unsigned char>(s[i]))) {
        case 'K': value <<= 10; break;
        case 'M': value <<= 20; break;
        case 'G': value <<= 30; break;
      }
    }
    return value;
  };

  auto read = [](const std::string& path) {
    auto s = read_file_to_string(path);
    if (!s.has_value())
      return std::string();
    // 末尾の改行などを取り除く
    while (!s->empty() && std::isspace(static_cast<unsigned char>(s->back())))
      s->pop_back();
    return *s;
  };

  for (const auto cpu : parse_cpu_list(read("/sys/devices/system/cpu/online"))) {
    const std::string cache_dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/";
    CacheLevels levels{};
    for (int index = 0;; ++index) {
      const std::string dir = cache_dir + "index" + std::to_string(index) + "/";
      const std::string level_str = read(dir + "level");
      if (level_str.empty())
        break;
      const int level = std::atoi(level_str.c_str());
      const std::string type = read(dir + "type");
      // パラメーターはデータとして読まれるので命令キャッシュは対象外。
      if (level < 1 || level > kCacheLevels || type == "Instruction")
        continue;
      levels[level - 1].size        = parse_size(read(dir + "size"));
      levels[level - 1].shared_cpus = parse_cpu_list(read(dir + "shared_cpu_list")).size();
    }
    per_cpu[cpu] = levels;
  }

#elif defined(_WIN64)

  DWORD buf_size = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &buf_size);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    return per_cpu;

  std::vector<char> buffer(buf_size);
  auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationCache, info, &buf_size))
    return per_cpu;

  while (reinterpret_cast<char*>(info) < buffer.data() + buf_size) {
    info = std::launder(info);
    const int level = info->Cache.Level;
    // パラメーターはデータとして読まれるので命令キャッシュは対象外。
    if (info->Relationship == RelationCache && level >= 1 && level <= kCacheLevels
        && info->Cache.Type != CacheInstruction) {
      const auto cpus = readCacheMembers(info, [](CpuIndex) { return true; });
      for (const auto cpu : cpus) {
        per_cpu[cpu][level - 1].size        = info->Cache.CacheSize;
        per_cpu[cpu][level - 1].shared_cpus = cpus.size();
      }
    }
    // 可変長のデータ構造なので、Sizeだけ進める
    info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
        reinterpret_cast<char*>(info) + info->Size);
  }

#endif

  return per_cpu;
}

// キャッシュ構成が同じ論理CPUごとにまとめる。(P-core/E-coreが混在するCPU対策)
std::vector<CacheGroup> DetectCacheGroups() {
  std::map<CacheLevels, std::vector<std::size_t>> grouped;
  for (const auto& [cpu, levels] : DetectCachePerCpu())
    grouped[levels].push_back(cpu);

  std::vector<CacheGroup> groups;
  for (auto& [levels, cpus] : grouped)
    groups.push_back(CacheGroup{levels, std::move(cpus)});
  return groups;
}

// "0-7,16-23" のような表記にする。
std::string CpuListString(const std::vector<std::size_t>& cpus) {
  std::string s;
  for (std::size_t i = 0; i < cpus.size();) {
    std::size_t j = i;
    while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1)
      ++j;
    if (!s.empty())
      s += ",";
    s += std::to_string(cpus[i]);
    if (j > i)
      s += "-" + std::to_string(cpus[j]);
    i = j + 1;
  }
  return s;
}

// bytesをKiB/MiBで表記する。
std::string SizeString(double bytes) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);
  if (bytes < 1024.0 * 1024.0)
    ss << bytes / 1024.0 << " KiB";
  else
    ss << bytes / (1024.0 * 1024.0) << " MiB";
  return ss.str();
}

// 表示する1行分。nameの先頭の空白は字下げとして使う。
struct SizeRow {
  std::string name;
  std::size_t bytes;
};

// ---- 層ごとのサイズ ----
// 各層のクラスが確保しているメモリ量(alignmentによるpaddingを含む)を求める。
// 活性化層など、パラメーターを持たない層は0 bytesとする。

template <typename T>
struct LayerTag {};

// 知らない層。(sizeofをそのまま使う)
template <typename T>
void AppendLayers(LayerTag<T>, std::vector<SizeRow>& rows, const std::string& indent) {
  rows.push_back({indent + "unknown layer", sizeof(T)});
}

template <IndexType OutputDimensions, IndexType Offset>
void AppendLayers(LayerTag<Layers::InputSlice<OutputDimensions, Offset>>, std::vector<SizeRow>& rows,
                  const std::string& indent) {
  rows.push_back({indent + "InputSlice[" + std::to_string(OutputDimensions) + "]", 0});
}

template <typename PreviousLayer>
void AppendLayers(LayerTag<Layers::ClippedReLU<PreviousLayer>>, std::vector<SizeRow>& rows,
                  const std::string& indent) {
  using L = Layers::ClippedReLU<PreviousLayer>;
  AppendLayers(LayerTag<PreviousLayer>{}, rows, indent);
  rows.push_back({indent + "ClippedReLU[" + std::to_string(L::kOutputDimensions) + "]", 0});
}

template <typename PreviousLayer, IndexType OutputDimensions>
void AppendLayers(LayerTag<Layers::AffineTransform<PreviousLayer, OutputDimensions>>,
                  std::vector<SizeRow>& rows, const std::string& indent) {
  using L = Layers::AffineTransform<PreviousLayer, OutputDimensions>;
  AppendLayers(LayerTag<PreviousLayer>{}, rows, indent);
  rows.push_back({indent + "AffineTransform[" + std::to_string(L::kInputDimensions) + "->"
                      + std::to_string(L::kOutputDimensions) + "]",
                  sizeof(L) - sizeof(PreviousLayer)});
}

template <typename PreviousLayer, IndexType OutputDimensions>
void AppendLayers(LayerTag<Layers::AffineTransformSparseInput<PreviousLayer, OutputDimensions>>,
                  std::vector<SizeRow>& rows, const std::string& indent) {
  using L = Layers::AffineTransformSparseInput<PreviousLayer, OutputDimensions>;
  AppendLayers(LayerTag<PreviousLayer>{}, rows, indent);
  rows.push_back({indent + "AffineTransformSparseInput[" + std::to_string(L::kInputDimensions) + "->"
                      + std::to_string(L::kOutputDimensions) + "]",
                  sizeof(L) - sizeof(PreviousLayer)});
}

#if defined(SFNNwoPSQT)

// SFNNの各層。メンバー名と型から1行を作る。
template <typename T>
SizeRow SfnnLayerRow(const std::string& indent, const std::string& member) {
  return {indent + member + " " + T::GetStructureString(), sizeof(T)};
}

template <typename T, typename = void>
struct IsSfnnAffine16: std::false_type {};
template <typename T>
struct IsSfnnAffine16<T, std::void_t<decltype(T::PaddedInputs)>>: std::true_type {};

template <typename T>
SizeRow SfnnAffineRow(const std::string& indent, const std::string& member) {
  if constexpr (IsSfnnAffine16<T>::value) {
    // SfnnAffine16は次元数を公開していないので、配列の大きさから求める。
    constexpr IndexType outputs = sizeof(T::biases) / sizeof(T::biases[0]);
    return {indent + member + " Affine16[" + std::to_string(outputs) + "<-"
                + std::to_string(T::PaddedInputs) + "(padded)]",
            sizeof(T)};
  } else {
    return SfnnLayerRow<T>(indent, member);
  }
}

void AppendSfnnLayers(std::vector<SizeRow>& rows, const std::string& indent) {
  using Fc0 = decltype(std::declval<Network&>().fc_0);
  using Fc1 = decltype(std::declval<Network&>().fc_1);
  using Fc2 = decltype(std::declval<Network&>().fc_2);
  const std::string h1 = std::to_string(Network::kHidden1Dims);
  const std::string h2 = std::to_string(Network::kHidden2Dims);

  std::size_t sum = sizeof(Fc0) + sizeof(Fc1) + sizeof(Fc2);
  rows.push_back(SfnnAffineRow<Fc0>(indent, "fc_0"));
#if defined(ENABLE_SFNN_16BIT_WEIGHT)
  // nn16では活性化層はメンバーとして持たず、関数で計算している。
  rows.push_back({indent + "ac_0 ClippedReLU[" + h1 + "]", 0});
  rows.push_back({indent + "ac_sqr_0 SqrClippedReLU[" + h1 + "]", 0});
  rows.push_back(SfnnAffineRow<Fc1>(indent, "fc_1"));
  rows.push_back({indent + "ac_1 ClippedReLU[" + h2 + "]", 0});
#else
  using Ac0    = decltype(std::declval<Network&>().ac_0);
  using AcSqr0 = decltype(std::declval<Network&>().ac_sqr_0);
  using Ac1    = decltype(std::declval<Network&>().ac_1);
  // 活性化層はパラメーターを持たない(空のclassなのでsizeofは1になる)。
  rows.push_back({indent + "ac_0 ClippedReLU[" + h1 + "]", 0});
  rows.push_back({indent + "ac_sqr_0 SqrClippedReLU[" + h1 + "]", 0});
  rows.push_back(SfnnAffineRow<Fc1>(indent, "fc_1"));
  rows.push_back({indent + "ac_1 ClippedReLU[" + h2 + "]", 0});
  sum += sizeof(Ac0) + sizeof(AcSqr0) + sizeof(Ac1);
#endif
  rows.push_back(SfnnAffineRow<Fc2>(indent, "fc_2"));

  // 層以外のメンバーやalignmentによるpadding
  if (sizeof(Network) > sum)
    rows.push_back({indent + "other members / padding", sizeof(Network) - sum});
}

#endif // defined(SFNNwoPSQT)

void AppendNetworkLayers(std::vector<SizeRow>& rows, const std::string& indent) {
#if defined(SFNNwoPSQT)
  AppendSfnnLayers(rows, indent);
#else
  AppendLayers(LayerTag<Network>{}, rows, indent);
#endif
}

// 現在の局面で、FeatureTransformerの重みのうち参照される行の数を数える。
// (差分計算ではなく全計算をしたときに読まれる行)
std::size_t CountActiveFeatureRows(const Position& pos) {
  std::set<IndexType> rows;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    Features::IndexList active_indices[2];
    RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
    for (const auto perspective : COLOR)
      for (const auto index : active_indices[perspective])
        rows.insert(index);
  }
  return rows.size();
}

void PrintCacheFit(IEngine& engine) {
  auto& options = engine.get_options();
  const std::size_t threads =
      options.count("Threads") ? std::max<std::size_t>(std::size_t(int64_t(options["Threads"])), 1) : 1;

  // ---- パラメーターのサイズ ----

  using FT = FeatureTransformer;
  constexpr std::size_t ft_bias_bytes = kTransformedFeatureDimensions * sizeof(FT::BiasType);
  constexpr std::size_t ft_row_bytes  = kTransformedFeatureDimensions * sizeof(FT::WeightType);
  constexpr std::size_t ft_max_rows   = std::size_t(RawFeatures::kMaxActiveDimensions) * 2;
  const std::size_t ft_cur_rows       = CountActiveFeatureRows(engine.get_position());

  // キャッシュと比較する項目
  struct Item {
    std::string name;
    std::size_t bytes;
  };
  std::vector<Item> items;

  std::vector<SizeRow> rows;
  rows.push_back({"FeatureTransformer (total) " + FT::GetStructureString(), sizeof(FT)});
  rows.push_back({"  biases", ft_bias_bytes});
  rows.push_back({"  weights (" + std::to_string(FT::kInputDimensions) + " rows)",
                  ft_row_bytes * FT::kInputDimensions});
  rows.push_back({"  one feature row", ft_row_bytes});
  rows.push_back({"FeatureTransformer touched, this position (biases + "
                      + std::to_string(ft_cur_rows) + " rows)",
                  ft_bias_bytes + ft_row_bytes * ft_cur_rows});
  rows.push_back({"FeatureTransformer touched, max (biases + " + std::to_string(ft_max_rows) + " rows)",
                  ft_bias_bytes + ft_row_bytes * ft_max_rows});
  rows.push_back({"Network, 1 stack", sizeof(Network)});
  AppendNetworkLayers(rows, "  ");
  if (kLayerStacks > 1)
    rows.push_back({"Network, all " + std::to_string(kLayerStacks) + " stacks",
                    sizeof(Network) * kLayerStacks});
#if defined(SFNNwoPSQT) && NNUE_SFNN_PROGRESS_BUCKETS != 1
  rows.push_back({"Progress parameters (progress.bin)", sizeof(Progress::Parameters)});
#endif
  rows.push_back({"All parameters (total)", sizeof(NnueNetworks)});

  // 字下げのない行をキャッシュとの比較対象にする。
  for (const auto& row : rows)
    if (row.name.empty() || row.name[0] != ' ')
      items.push_back({row.name, row.bytes});

  std::size_t name_width = 0;
  for (const auto& row : rows)
    name_width = std::max(name_width, row.name.size());

  std::cout << "network architecture: " << GetArchitectureString() << std::endl;
  std::cout << "Threads: " << threads << std::endl;
  std::cout << std::endl << "[parameter sizes]" << std::endl;
  for (const auto& row : rows)
    std::cout << "  " << std::left << std::setw(int(name_width)) << row.name << "  " << std::right
              << std::setw(12) << SizeString(double(row.bytes)) << std::endl;

  // ---- キャッシュ ----

  const auto groups = DetectCacheGroups();
  if (groups.empty()) {
    std::cout << std::endl
              << "cache sizes could not be detected on this platform." << std::endl;
    return;
  }

  // 3通りの基準でのキャッシュ容量
  const char* const basis_names[] = {"total", "per sharing CPU", "per thread"};

  for (std::size_t g = 0; g < groups.size(); ++g) {
    const auto& group = groups[g];
    std::cout << std::endl
              << "[cache group " << (g + 1) << "] logical CPUs: " << CpuListString(group.cpus) << " ("
              << group.cpus.size() << " CPUs)" << std::endl;

    // キャッシュ容量の一覧
    std::cout << "  " << std::left << std::setw(6) << "level" << std::right << std::setw(14) << "total"
              << std::setw(12) << "shared by" << std::setw(18) << "per sharing CPU" << std::setw(16)
              << "per thread" << std::endl;
    for (int l = 0; l < kCacheLevels; ++l) {
      const auto& c = group.levels[l];
      std::cout << "  " << std::left << std::setw(6) << kCacheLevelNames[l] << std::right;
      if (c.size == 0) {
        std::cout << std::setw(14) << "n/a" << std::endl;
        continue;
      }
      const std::size_t shared = std::max<std::size_t>(c.shared_cpus, 1);
      std::cout << std::setw(14) << SizeString(double(c.size)) << std::setw(12)
                << (std::to_string(shared) + " CPUs") << std::setw(18)
                << SizeString(double(c.size) / shared) << std::setw(16)
                << SizeString(double(c.size) / threads) << std::endl;
    }

    // 各項目が載るかの判定。占有率 = 項目のサイズ / キャッシュ容量
    std::size_t item_width = 0;
    for (const auto& item : items)
      item_width = std::max(item_width, item.name.size());

    std::cout << std::endl << "  " << std::left << std::setw(int(item_width)) << "item" << "  "
              << std::setw(16) << "basis";
    for (int l = 0; l < kCacheLevels; ++l)
      std::cout << std::setw(16) << kCacheLevelNames[l];
    std::cout << std::right << std::endl;

    for (const auto& item : items) {
      for (int b = 0; b < 3; ++b) {
        std::cout << "  " << std::left << std::setw(int(item_width)) << (b == 0 ? item.name : "")
                  << "  " << std::setw(16) << basis_names[b];
        for (int l = 0; l < kCacheLevels; ++l) {
          const auto& c = group.levels[l];
          if (c.size == 0) {
            std::cout << std::setw(16) << "n/a";
            continue;
          }
          const double divisor = b == 0 ? 1.0
                               : b == 1 ? double(std::max<std::size_t>(c.shared_cpus, 1))
                                        : double(threads);
          const double capacity = double(c.size) / divisor;
          const double ratio    = 100.0 * double(item.bytes) / capacity;
          std::ostringstream cell;
          cell << (double(item.bytes) <= capacity ? "yes " : "no  ") << std::fixed
               << std::setprecision(1) << ratio << "%";
          std::cout << std::setw(16) << cell.str();
        }
        std::cout << std::right << std::endl;
      }
    }
  }
}

}  // namespace

// NNUE評価関数に関するUSI拡張コマンド
void TestCommand(IEngine& engine, std::istream& stream) {
  std::string sub_command;
  stream >> sub_command;

  if (sub_command == "test_features") {
    TestFeatures(engine.get_position());
  } else if (sub_command == "info") {
    PrintInfo(stream);
  } else if (sub_command == "cache") {
    PrintCacheFit(engine);
  } else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nn test_features" << std::endl;
    std::cout << " test nn info [path/to/" << kFileName << "...]" << std::endl;
    std::cout << " test nn cache" << std::endl;
  }
}

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)
