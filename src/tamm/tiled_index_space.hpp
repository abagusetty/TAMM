#ifndef TAMM_TILED_INDEX_SPACE_HPP_
#define TAMM_TILED_INDEX_SPACE_HPP_

#include "tamm/index_space.hpp"

namespace tamm {

class TiledIndexLabel;

/**
 * @brief TiledIndexSpace class
 *
 */
class TiledIndexSpace {
public:
    TiledIndexSpace()                       = default;
    TiledIndexSpace(const TiledIndexSpace&) = default;
    TiledIndexSpace(TiledIndexSpace&&)      = default;
    TiledIndexSpace& operator=(TiledIndexSpace&&) = default;
    TiledIndexSpace& operator=(const TiledIndexSpace&) = default;
    ~TiledIndexSpace()                                 = default;

    /**
     * @brief Construct a new TiledIndexSpace from
     * a reference IndexSpace and a tile size
     *
     * @param [in] is reference IndexSpace
     * @param [in] input_tile_size tile size (default: 1)
     */
    TiledIndexSpace(const IndexSpace& is, Tile input_tile_size = 1) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        is, input_tile_size, std::vector<Tile>{})},
      root_tiled_info_{tiled_info_},
      parent_tis_{nullptr} {
        EXPECTS(input_tile_size > 0);
        compute_hash();
        // construct tiling for named subspaces
        tile_named_subspaces(is);
    }

    /**
     * @brief Construct a new TiledIndexSpace from a reference
     * IndexSpace and varying tile sizes
     *
     * @param [in] is
     * @param [in] input_tile_sizes
     */
    TiledIndexSpace(const IndexSpace& is,
                    const std::vector<Tile>& input_tile_sizes) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        is, 0, input_tile_sizes)},
      root_tiled_info_{tiled_info_},
      parent_tis_{nullptr} {
        for(const auto& in_tsize : input_tile_sizes) { EXPECTS(in_tsize > 0); }
        compute_hash();
        // construct tiling for named subspaces
        tile_named_subspaces(is);
    }

    TiledIndexSpace(
      const IndexSpace& is,
      const std::map<IndexVector, std::vector<Tile>>& dep_tile_sizes) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        is, dep_tile_sizes)},
      root_tiled_info_{tiled_info_},
      parent_tis_{nullptr} {
        for(const auto& [idx_vec, tile_sizes] : dep_tile_sizes) {
            EXPECTS(tile_sizes.size() > 0);
        }
        compute_hash();
        // construct tiling for named subspaces
        // tile_named_subspaces(is);
    }

    /**
     * @brief Construct a new sub TiledIndexSpace object from
     * a sub-space of a reference TiledIndexSpace
     *=
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] range Range of the reference TiledIndexSpace
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const Range& range) :
      TiledIndexSpace{t_is, construct_index_vector(range)} {}

    /**
     * @brief Construct a new sub TiledIndexSpace object from
     * a sub-space of a reference TiledIndexSpace
     *
     * @param [in] t_is reference TiledIndexSpace
     * @param [in] indices set of indices of the reference TiledIndexSpace
     */
    TiledIndexSpace(const TiledIndexSpace& t_is, const IndexVector& indices) :
      root_tiled_info_{t_is.root_tiled_info_},
      parent_tis_{std::make_shared<TiledIndexSpace>(t_is)} {
        EXPECTS(parent_tis_ != nullptr);
        IndexVector new_indices, new_offsets;

        IndexVector is_indices;
        auto root_tis = parent_tis_;
        while(root_tis->parent_tis_ != nullptr) {
            root_tis = root_tis->parent_tis_;
        }

        new_offsets.push_back(0);
        for(const auto& idx : indices) {
            auto new_idx = t_is.info_translate(idx, (*root_tiled_info_.lock()));
            new_indices.push_back(new_idx);
            new_offsets.push_back(new_offsets.back() +
                                  root_tiled_info_.lock()->tile_size(new_idx));

            for(auto i = root_tis->block_begin(new_idx);
                i != root_tis->block_end(new_idx); i++) {
                is_indices.push_back((*i));
            }
        }

        IndexSpace sub_is{root_tis->index_space(), is_indices};

        tiled_info_ = std::make_shared<TiledIndexSpaceInfo>(
          (*t_is.root_tiled_info_.lock()), sub_is, new_offsets, new_indices);
        compute_hash();
    }

    /**
     * @brief Construct a new Tiled Index Space object from a tiled dependent
     *        index space
     *
     * @param [in] t_is parent tiled index space
     * @param [in] dep_map dependency map
     */
    TiledIndexSpace(const TiledIndexSpace& t_is,
                    const TiledIndexSpaceVec& dep_vec,
                    const std::map<IndexVector, TiledIndexSpace>& dep_map) :
      tiled_info_{std::make_shared<TiledIndexSpace::TiledIndexSpaceInfo>(
        (*t_is.tiled_info_), dep_vec, dep_map)},
      root_tiled_info_{t_is.root_tiled_info_},
      parent_tis_{std::make_shared<TiledIndexSpace>(t_is)} {
        // validate dependency map
        std::vector<std::size_t> tis_sizes;
        for(const auto& tis : dep_vec) { tis_sizes.push_back(tis.num_tiles()); }

        for(const auto& [key, value] : dep_map) {
            EXPECTS(key.size() == dep_vec.size());
            for(size_t i = 0; i < key.size(); i++) {
                EXPECTS(key[i] < tis_sizes[i]);
            }
            EXPECTS(value.is_subset_of(t_is));
        }

        compute_hash();
    }

    /**
     * @brief Get a TiledIndexLabel for a specific subspace of the
     * TiledIndexSpace
     *
     * @param [in] id string name for the subspace
     * @param [in] lbl an integer value for associated Label
     * @returns a TiledIndexLabel associated with a TiledIndexSpace
     */
    TiledIndexLabel label(std::string id, Label lbl = Label{0}) const;

    TiledIndexLabel label(Label lbl = Label{0}) const;

    /**
     * @brief Construct a tuple of TiledIndexLabel given a count, subspace name
     * and a starting integer Label
     *
     * @tparam c_lbl count of labels
     * @param [in] id name string associated to the subspace
     * @param [in] start starting label value
     * @returns a tuple of TiledIndexLabel
     */
    template<std::size_t c_lbl>
    auto labels(std::string id, Label start = Label{0}) const {
        return labels_impl(id, start, std::make_index_sequence<c_lbl>{});
    }

    /**
     * @brief operator () overload for accessing a (sub)TiledIndexSpace with the
     * given subspace name string
     *
     * @param [in] id name string associated to the subspace
     * @returns a (sub)TiledIndexSpace associated with the subspace name string
     */
    TiledIndexSpace operator()(std::string id) const {
        if(id == "all") { return (*this); }

        return tiled_info_->tiled_named_subspaces_.at(id);
    }

    /**
     * @brief Operator overload for getting TiledIndexSpace from dependent
     * relation map
     *
     * @param [in] dep_idx_vec set of dependent index values
     * @returns TiledIndexSpace from the relation map
     */
    TiledIndexSpace operator()(const IndexVector& dep_idx_vec = {}) const {
        if(dep_idx_vec.empty()) { return (*this); }
        const auto& t_dep_map = tiled_info_->tiled_dep_map_;
        EXPECTS(t_dep_map.find(dep_idx_vec) != t_dep_map.end());
        // if(t_dep_map.find(dep_idx_vec) == t_dep_map.end()){
        //     return TiledIndexSpace{IndexSpace{{}}};
        // }

        return t_dep_map.at(dep_idx_vec);
    }

    /**
     * @brief Iterator accessor to the start of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the first element of the
     * IndexSpace
     */
    IndexIterator begin() const { return tiled_info_->simple_vec_.begin(); }

    /**
     * @brief Iterator accessor to the end of the reference IndexSpace
     *
     * @returns a const_iterator to an Index at the size-th element of the
     * IndexSpace
     */
    IndexIterator end() const { return tiled_info_->simple_vec_.end(); }

    /**
     * @brief Iterator accessor to the first Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the first Index element of the specific
     * block
     */
    IndexIterator block_begin(Index blck_ind) const {
        EXPECTS(blck_ind <= num_tiles());
        return tiled_info_->is_.begin() + tiled_info_->tile_offsets_[blck_ind];
    }
    /**
     * @brief Iterator accessor to the last Index element of a specific block
     *
     * @param [in] blck_ind Index of the block to get const_iterator
     * @returns a const_iterator to the last Index element of the specific
     * block
     */
    IndexIterator block_end(Index blck_ind) const {
        EXPECTS(blck_ind <= num_tiles());
        return tiled_info_->is_.begin() +
               tiled_info_->tile_offsets_[blck_ind + 1];
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is identical
     * to this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the tiled_info_ pointer is the same for both
     * TiledIndexSpaces
     */
    bool is_identical(const TiledIndexSpace& rhs) const {
        return (hash_value_ == rhs.hash());
    }

    /**
     * @brief Boolean method for checking if this TiledIndexSpace is a subset of
     * input TiledIndexSpace
     *
     * @param [in] tis reference TiledIndexSpace
     * @returns true if this is a subset of input TiledIndexSpace
     */
    bool is_subset_of(const TiledIndexSpace& tis) const {
        if(this->is_identical(tis)) {
            return true;
        } else if(this->is_dependent()) {
            if(tis.is_dependent()) {
                return (this->parent_tis_->is_subset_of((*tis.parent_tis_)));
            } else {
                return (this->parent_tis_->is_subset_of(tis));
            }

        } else if(this->parent_tis_ != nullptr) {
            return (this->parent_tis_->is_subset_of(tis));
        }

        return false;
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace is compatible
     * to this TiledIndexSpace
     *
     * @param [in] tis reference TiledIndexSpace
     * @returns true if the root_tiled_info_ is the same for both
     * TiledIndexSpaces
     */
    bool is_compatible_with(const TiledIndexSpace& tis) const {
        return is_subset_of(tis);
    }

    /**
     * @brief Boolean method for checking if given TiledIndexSpace
     * is a subspace of this TiledIndexSpace
     *
     * @param [in] rhs reference TiledIndexSpace
     * @returns true if the TiledIndexInfo object of rhs is constructed later
     * then this
     */
    bool is_less_than(const TiledIndexSpace& rhs) const {
        return (tiled_info_ < rhs.tiled_info_);
    }

    /**
     * @brief Accessor methods to Spin value associated with the input Index
     *
     * @param [in] idx input Index value
     * @returns associated Spin value for the input Index value
     */
    Spin spin(size_t idx) const {
        size_t translated_idx = info_translate(idx, (*root_tiled_info_.lock()));
        return root_tiled_info_.lock()->spin_value(translated_idx);
    }

    /**
     * @brief Accessor methods to Spatial value associated with the input Index
     *
     * @todo: fix once we have spatial
     *
     * @param [in] idx input Index value
     * @returns associated Spatial value for the input Index value
     */
    Spatial spatial(size_t idx) const { return tiled_info_->is_.spatial(idx); }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spin value
     *
     * @param [in] spin input Spin value
     * @returns a vector of Ranges associated with the input Spin value
     */
    std::vector<Range> spin_ranges(Spin spin) const {
        return tiled_info_->is_.spin_ranges(spin);
    }

    /**
     * @brief Accessor method for the set of Ranges associated with a Spatial
     * value
     *
     * @param [in] spatial input Spatial value
     * @returns a vector of Ranges associated with the input Spatial value
     */
    std::vector<Range> spatial_ranges(Spatial spatial) const {
        return tiled_info_->is_.spatial_ranges(spatial);
    }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpinAttribute
     *
     * @returns true if there is a SpinAttribute associated with the IndexSpace
     */
    bool has_spin() const { return tiled_info_->is_.has_spin(); }

    /**
     * @brief Boolean method for checking if an IndexSpace has SpatialAttribute
     *
     * @return true if there is a SpatialAttribute associated with the
     * IndexSpace
     */
    bool has_spatial() const { return tiled_info_->is_.has_spatial(); }

    /**
     * @brief Getter method for the reference IndexSpace
     *
     * @return IndexSpace reference
     */
    const IndexSpace& index_space() const { return tiled_info_->is_; }

    /**
     * @brief Get the number of tiled index blocks in TiledIndexSpace
     *
     * @return number of tiles in the TiledIndexSpace
     */
    std::size_t num_tiles() const {
        if(is_dependent()) { NOT_ALLOWED(); }

        return tiled_info_->tile_offsets_.size() - 1;
    }

    const IndexVector& ref_indices() const { return tiled_info_->ref_indices_; }
    /**
     * @brief Get the maximum number of tiled index blocks in TiledIndexSpace
     *
     * @return maximum number of tiles in the TiledIndexSpace
     */
    std::size_t max_num_tiles() const { return tiled_info_->max_num_tiles(); }

    /**
     * @brief Get the tile size for the index blocks
     *
     * @return Tile size
     */
    std::size_t tile_size(Index i) const { return tiled_info_->tile_size(i); }

    /**
     * @brief Get the input tile size for tiled index space
     *
     * @returns input tile size
     */
    Tile input_tile_size() const { return tiled_info_->input_tile_size_; }

    /**
     * @brief Get the input tile size for tiled index space
     *
     * @returns input tile sizes
     */
    const std::vector<Tile>& input_tile_sizes() const {
        return tiled_info_->input_tile_sizes_;
    }

    /**
     * @brief Get tiled dependent spaces map
     *
     * @returns a map from dependent indicies to tiled index spaces
     */
    const std::map<IndexVector, TiledIndexSpace>& tiled_dep_map() const {
        return tiled_info_->tiled_dep_map_;
    }

    /**
     * @brief Accessor to tile offsets
     *
     * @return Tile offsets
     */
    const IndexVector& tile_offsets() const {
        return tiled_info_->tile_offsets_;
    }

    /**
     * @brief Accessor to tile offset with index id
     *
     * @param [in] id index for the tile offset
     * @returns offset for the corresponding index
     */
    const std::size_t tile_offset(size_t id) const {
        EXPECTS(id >= 0 && id < tiled_info_->simple_vec_.size());

        return tile_offsets()[id];
    }

    /**
     * @brief Translate id to another tiled index space
     *
     * @param [in] id index to be translated
     * @param [in] new_tis reference index space to translate to
     * @returns an index from the new_tis that corresponds to [in] id
     */
    std::size_t translate(size_t id, const TiledIndexSpace& new_tis) const {
        EXPECTS(!is_dependent());
        EXPECTS(id >= 0 && id < tiled_info_->ref_indices_.size());
        if(new_tis == (*this)) { return id; }

        auto new_ref_indices = new_tis.tiled_info_->ref_indices_;
        EXPECTS(new_ref_indices.size() ==
                new_tis.tiled_info_->simple_vec_.size());

        auto it = std::find(new_ref_indices.begin(), new_ref_indices.end(),
                            tiled_info_->ref_indices_[id]);
        EXPECTS(it != new_ref_indices.end());

        return (it - new_ref_indices.begin());
    }

    /**
     * @brief Check if reference index space is a dependent index space
     *
     * @returns true if the reference index space is a dependent index space
     */
    const bool is_dependent() const {
        // return tiled_info_->is_.is_dependent();
        return !tiled_info_->tiled_dep_map_.empty();
    }

    size_t hash() const { return hash_value_; }

    size_t num_key_tiled_index_spaces() const {
        if(tiled_info_->is_.is_dependent()) {
            return tiled_info_->is_.num_key_tiled_index_spaces();
        }
        return tiled_info_->dep_vec_.size();
    }

    /**
     * @brief Equality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs == rhs
     */
    friend bool operator==(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs < rhs
     */
    friend bool operator<(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);

    /**
     * @brief Inequality comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs != rhs
     */
    friend bool operator!=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs > rhs
     */
    friend bool operator>(const TiledIndexSpace& lhs,
                          const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs <= rhs
     */
    friend bool operator<=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

    /**
     * @brief Ordering comparison operator
     *
     * @param [in] lhs Left-hand side
     * @param [in] rhs Right-hand side
     *
     * @return true if lhs >= rhs
     */
    friend bool operator>=(const TiledIndexSpace& lhs,
                           const TiledIndexSpace& rhs);

protected:
    /**
     * @brief Internal struct for representing a TiledIndexSpace details. Mainly
     * used for comparing TiledIndexSpace between eachother for compatibility
     * checks. This also behaves as PIMPL to ease the copy of TiledIndexSpaces.
     *
     */
    struct TiledIndexSpaceInfo {
        /* data */
        IndexSpace is_;        /**< The index space being tiled*/
        Tile input_tile_size_; /**< User-specified tile size*/
        std::vector<Tile>
          input_tile_sizes_; /**< User-specified multiple tile sizes*/
        std::map<IndexVector, std::vector<Tile>> dep_tile_sizes_;
        IndexVector tile_offsets_; /**< Tile offsets */
        IndexVector ref_indices_;  /**< Reference indices to root */
        IndexVector simple_vec_;   /**< vector where at(i) = i*/
        std::size_t
          max_num_tiles_; /**<Maximum number of tiles in this tiled space*/
        TiledIndexSpaceVec dep_vec_;
        std::map<IndexVector, TiledIndexSpace>
          tiled_dep_map_; /**< Tiled dependency map for dependent index spaces*/
        std::map<std::string, TiledIndexSpace>
          tiled_named_subspaces_; /**< Tiled named subspaces map string ids*/

        /**
         * @brief Construct a new TiledIndexSpaceInfo object from an IndexSpace
         * and input tile size(s). The size can be a single size or a set of
         * tile sizes. Note that, set of tiles sizes are required to tile the
         * underlying IndexSpace completely.
         *
         * @param [in] is reference IndexSpace to be tiled
         * @param [in] input_tile_size input single tile size
         * @param [in] input_tile_sizes input set of tile sizes
         */
        TiledIndexSpaceInfo(IndexSpace is, Tile input_tile_size,
                            const std::vector<Tile>& input_tile_sizes) :
          is_{is},
          input_tile_size_{input_tile_size},
          input_tile_sizes_{input_tile_sizes} {
            if(input_tile_sizes.size() > 0) {
                // construct indices with set of tile sizes
                tile_offsets_ = construct_tiled_indices(is, input_tile_sizes);
                // construct dependency according to tile sizes
                for(const auto& kv : is.map_tiled_index_spaces()) {
                    tiled_dep_map_.insert(
                      std::pair<IndexVector, TiledIndexSpace>{
                        kv.first,
                        TiledIndexSpace{kv.second, input_tile_sizes}});
                }
                // in case of multiple tile sizes no named spacing carried.
            } else {
                // construct indices with input tile size
                tile_offsets_ = construct_tiled_indices(is, input_tile_size);
                // construct dependency according to tile size
                for(const auto& kv : is.map_tiled_index_spaces()) {
                    tiled_dep_map_.insert(
                      std::pair<IndexVector, TiledIndexSpace>{
                        kv.first, TiledIndexSpace{kv.second, input_tile_size}});
                }
            }

            if(!is.is_dependent()) {
                for(Index i = 0; i < tile_offsets_.size() - 1; i++) {
                    simple_vec_.push_back(i);
                    ref_indices_.push_back(i);
                }
            }
            compute_max_num_tiles();
            validate();
        }

        TiledIndexSpaceInfo(
          IndexSpace is,
          std::map<IndexVector, std::vector<Tile>> dep_tile_sizes) :
          is_{is},
          input_tile_size_{0},
          input_tile_sizes_{{}},
          dep_tile_sizes_{dep_tile_sizes} {
            EXPECTS(is.is_dependent());
            // construct dependency according to tile size
            for(const auto& kv : is.map_tiled_index_spaces()) {
                EXPECTS(dep_tile_sizes.find(kv.first) != dep_tile_sizes.end());

                tiled_dep_map_.insert(std::pair<IndexVector, TiledIndexSpace>{
                  kv.first,
                  TiledIndexSpace{kv.second, dep_tile_sizes[kv.first]}});
            }

            compute_max_num_tiles();
            validate();
        }

        /**
         * @brief Construct a new sub-TiledIndexSpaceInfo object from a root
         * TiledIndexInfo object along with a set of offsets and indices
         * corresponding to the root object
         *
         * @param [in] root TiledIndexSpaceInfo object
         * @param [in] offsets input offsets from the root
         * @param [in] indices input indices from the root
         */
        TiledIndexSpaceInfo(const TiledIndexSpaceInfo& root,
                            const IndexVector& offsets,
                            const IndexVector& indices) :
          is_{root.is_},
          input_tile_size_{root.input_tile_size_},
          input_tile_sizes_{root.input_tile_sizes_},
          tile_offsets_{offsets},
          ref_indices_{indices} {
            for(Index i = 0; i < tile_offsets_.size() - 1; i++) {
                simple_vec_.push_back(i);
            }
            compute_max_num_tiles();
            validate();
        }

        TiledIndexSpaceInfo(const TiledIndexSpaceInfo& root,
                            const IndexSpace& is, const IndexVector& offsets,
                            const IndexVector& indices) :
          is_{is},
          input_tile_size_{root.input_tile_size_},
          input_tile_sizes_{root.input_tile_sizes_},
          tile_offsets_{offsets},
          ref_indices_{indices} {
            for(Index i = 0; i < tile_offsets_.size() - 1; i++) {
                simple_vec_.push_back(i);
            }
            compute_max_num_tiles();
            validate();
        }

        /**
         * @brief Construct a new TiledIndexSpaceInfo object from a root
         * (dependent) TiledIndexSpaceInfo object with a new set of relations
         * for the dependency relation.
         *
         * @param [in] root TiledIndexSpaceInfo object
         * @param [in] dep_map dependency relation between indices of the
         * dependent TiledIndexSpace and corresponding TiledIndexSpaces
         */
        TiledIndexSpaceInfo(
          const TiledIndexSpaceInfo& root, const TiledIndexSpaceVec& dep_vec,
          const std::map<IndexVector, TiledIndexSpace>& dep_map) :
          is_{root.is_},
          input_tile_size_{root.input_tile_size_},
          input_tile_sizes_{root.input_tile_sizes_},
          dep_vec_{dep_vec},
          tiled_dep_map_{dep_map} {
            // Check if the new dependency relation is sub set of root
            // dependency relation
            // const auto& root_dep = root.tiled_dep_map_;
            // for(const auto& dep_kv : dep_map) {
            //     const auto& key     = dep_kv.first;
            //     const auto& dep_tis = dep_kv.second;
            //     EXPECTS(root_dep.find(key) != root_dep.end());
            //     EXPECTS(dep_tis.is_compatible_with(root_dep.at(key)));
            // }
            std::vector<IndexVector> result;
            IndexVector acc;
            combinations_tis(result, acc, dep_vec, 0);
            IndexSpace empty_is{IndexVector{}};
            TiledIndexSpace empty_tis{empty_is};

            for(const auto& iv : result) {
                if(tiled_dep_map_.find(iv) == tiled_dep_map_.end()){
                    tiled_dep_map_[iv] = empty_tis;
                }
            }

            compute_max_num_tiles();
            validate();
        }
        /**
         * @brief Construct starting and ending indices of each tile with
         * respect to input tile size.
         *
         * @param [in] is reference IndexSpace
         * @param [in] size Tile size value
         * @returns a vector of indices corresponding to the start and end of
         * each tile
         */
        IndexVector construct_tiled_indices(const IndexSpace& is,
                                            Tile tile_size) {
            if(is.is_dependent()) { return {}; }

            if(is.num_indices() == 0) { return {0}; }

            IndexVector boundries, ret;
            // Get lo and hi for each named subspace ranges
            for(const auto& kv : is.get_named_ranges()) {
                for(const auto& range : kv.second) {
                    boundries.push_back(range.lo());
                    boundries.push_back(range.hi());
                }
            }

            // Get SpinAttribute boundries
            if(is.has_spin()) {
                auto spin_map = is.get_spin().get_map();
                for(const auto& kv : spin_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }
            // Get SpinAttribute boundries
            if(is.has_spatial()) {
                auto spatial_map = is.get_spatial().get_map();
                for(const auto& kv : spatial_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            // If no boundry clean split with respect to tile size
            if(boundries.empty()) {
                // add starting indices
                for(size_t i = 0; i < is.num_indices(); i += tile_size) {
                    ret.push_back(i);
                }
                // add size of IndexSpace for the last block
                ret.push_back(is.num_indices());
            } else { // Remove duplicates
                std::sort(boundries.begin(), boundries.end());
                auto last = std::unique(boundries.begin(), boundries.end());
                boundries.erase(last, boundries.end());
                // Construct start indices for blocks according to boundries.
                std::size_t i = 0;
                std::size_t j = (i == boundries[0]) ? 1 : 0;

                while(i < is.num_indices()) {
                    ret.push_back(i);
                    i = (i + tile_size >= boundries[j]) ? boundries[j++] :
                                                          (i + tile_size);
                }
                // add size of IndexSpace for the last block
                ret.push_back(is.num_indices());
            }

            return ret;
        }

        /**
         * @brief Construct starting and ending indices of each tile with
         * respect to input tile sizes
         *
         * @param [in] is reference IndexSpace
         * @param [in] sizes set of input Tile sizes
         * @returns a vector of indices corresponding to the start and end of
         * each tile
         */
        IndexVector construct_tiled_indices(const IndexSpace& is,
                                            const std::vector<Tile>& tiles) {
            if(is.is_dependent()) { return {}; }

            if(is.num_indices() == 0) { return {0}; }
            // Check if sizes match
            EXPECTS(is.num_indices() == [&tiles]() {
                size_t ret = 0;
                for(const auto& var : tiles) { ret += var; }
                return ret;
            }());

            IndexVector ret, boundries;

            if(is.has_spin()) {
                auto spin_map = is.get_spin().get_map();
                for(const auto& kv : spin_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            if(is.has_spatial()) {
                auto spatial_map = is.get_spatial().get_map();
                for(const auto& kv : spatial_map) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            if(is.get_named_ranges().empty()) {
                for(const auto& kv : is.get_named_ranges()) {
                    for(const auto& range : kv.second) {
                        boundries.push_back(range.lo());
                        boundries.push_back(range.hi());
                    }
                }
            }

            // add starting indices
            size_t j = 0;
            for(size_t i = 0; i < is.num_indices(); i += tiles[j++]) {
                ret.push_back(i);
            }
            // add size of IndexSpace for the last block
            ret.push_back(is.num_indices());

            if(!(boundries.empty())) {
                std::sort(boundries.begin(), boundries.end());
                auto last = std::unique(boundries.begin(), boundries.end());
                boundries.erase(last, boundries.end());
                // check if there is any mismatch between boudries and generated
                // start indices
                for(auto& bound : boundries) {
                    EXPECTS(std::binary_search(ret.begin(), ret.end(), bound));
                }
            }

            return ret;
        }

        /**
         * @brief Accessor for getting the size of a specific tile in the
         * TiledIndexSpaceInfo object
         *
         * @param [in] idx input index
         * @returns the size of the tile at the corresponding index
         */
        std::size_t tile_size(Index idx) const {
            // std::cerr << "idx: " << idx << std::endl;
            EXPECTS(idx >= 0 && idx < tile_offsets_.size() - 1);
            return tile_offsets_[idx + 1] - tile_offsets_[idx];
        }

        /**
         * @brief Gets the maximum number of tiles in the TiledIndexSpaceInfo
         * object. In case of independent TiledIndexSpace it returns the number
         * of tiles, otherwise returns the maximum size of the TiledIndexSpaces
         * in the dependency relation
         *
         * @returns the maximum number of tiles in the TiledIndexSpaceInfo
         */
        std::size_t max_num_tiles() const { return max_num_tiles_; }

        void compute_max_num_tiles() {
            if(tiled_dep_map_.empty()) {
                max_num_tiles_ = tile_offsets_.size() - 1;
            } else {
                max_num_tiles_ = 0;
                for(const auto& kv : tiled_dep_map_) {
                    if(max_num_tiles_ < kv.second.max_num_tiles()) {
                        max_num_tiles_ = kv.second.max_num_tiles();
                    }
                }
            }
        }

        Spin spin_value(size_t id) const { return is_.spin(tile_offsets_[id]); }

        void validate() {
            // Post-condition
            EXPECTS(simple_vec_.size() == ref_indices_.size());
            if(tiled_dep_map_.empty()) {
                EXPECTS(simple_vec_.size() + 1 == tile_offsets_.size());
            }
        }

        void combinations_tis(std::vector<IndexVector>& res,
                              const IndexVector& accum,
                              const TiledIndexSpaceVec& tis_vec, size_t i) {
            if(i == tis_vec.size()) {
                res.push_back(accum);
            } else {
                auto tis = tis_vec[i];

                for(const auto& tile_id : tis) {
                    IndexVector tmp{accum};
                    tmp.push_back(tile_id);
                    combinations_tis(res, tmp, tis_vec, i + 1);
                }
            }
        }

        size_t hash() {
            // get hash of IndexSpace as seed
            size_t result = is_.hash();

            // combine hash with tile size(s)
            internal::hash_combine(result, input_tile_size_);
            // if there are mutliple tiles
            if(input_tile_sizes_.size() > 0) {
                // combine hash with number of tile sizes
                internal::hash_combine(result, input_tile_sizes_.size());
                // combine hash with each tile size
                for(const auto& tile : input_tile_sizes_) {
                    internal::hash_combine(result, tile);
                }
            }

            // if it is a dependent TiledIndexSpace
            if(!tiled_dep_map_.empty()) {
                for(const auto& iv_is : tiled_dep_map_) {
                    const auto& iv = iv_is.first;
                    const auto& is = iv_is.second;
                    // combine hash with size of index vector
                    internal::hash_combine(result, iv.size());
                    // combine hash with each key element
                    for(const auto& idx : iv) {
                        internal::hash_combine(result, idx);
                    }
                    // combine hash with dependent index space hash
                    internal::hash_combine(result, is.hash());
                }
            }

            /// @to-do add dep_tile_sizes stuff

            return result;
        }
    }; // struct TiledIndexSpaceInfo

    std::shared_ptr<TiledIndexSpaceInfo>
      tiled_info_; /**< Shared pointer to the TiledIndexSpaceInfo object*/
    std::weak_ptr<TiledIndexSpaceInfo>
      root_tiled_info_; /**< Weak pointer to the root TiledIndexSpaceInfo
                           object*/
    std::shared_ptr<TiledIndexSpace>
      parent_tis_; /**< Weak pointer to the parent
                    TiledIndexSpace object*/
    size_t hash_value_;

    /**
     * @brief Return the corresponding tile position of an index for a give
     * TiledIndexSpaceInfo object
     *
     * @param [in] id index position to be found on input TiledIndexSpaceInfo
     * object
     * @param [in] new_info reference input TiledIndexSpaceInfo object
     * @returns The tile index of the corresponding from the reference
     * TiledIndexSpaceInfo object
     */
    std::size_t info_translate(size_t id,
                               const TiledIndexSpaceInfo& new_info) const {
        EXPECTS(id >= 0 && id < tiled_info_->ref_indices_.size());
        EXPECTS(new_info.ref_indices_.size() == new_info.simple_vec_.size());

        auto it =
          std::find(new_info.ref_indices_.begin(), new_info.ref_indices_.end(),
                    tiled_info_->ref_indices_[id]);
        EXPECTS(it != new_info.ref_indices_.end());

        return (it - new_info.ref_indices_.begin());
    }

    /**
     * @brief Set the root TiledIndexSpaceInfo object
     *
     * @param [in] root a shared pointer to a TiledIndexSpaceInfo object
     */
    void set_root(const std::shared_ptr<TiledIndexSpaceInfo>& root) {
        root_tiled_info_ = root;
    }

    /**
     * @brief Set the parent TiledIndexSpace object
     *
     * @param [in] parent a shared pointer to a TiledIndexSpace object
     */
    void set_parent(const std::shared_ptr<TiledIndexSpace>& parent) {
        parent_tis_ = parent;
    }

    /**
     * @brief Set the shared pointer to TiledIndexSpaceInfo object
     *
     * @param [in] tiled_info shared pointer to TiledIndexSpaceInfo object
     */
    void set_tiled_info(
      const std::shared_ptr<TiledIndexSpaceInfo>& tiled_info) {
        tiled_info_ = tiled_info;
    }

    void compute_hash() { hash_value_ = tiled_info_->hash(); }

    /**
     * @brief Method for tiling all the named subspaces in an IndexSpace
     *
     * @param [in] is input IndexSpace
     */
    void tile_named_subspaces(const IndexSpace& is) {
        // construct tiled spaces for named subspaces
        for(const auto& str_subis : is.map_named_sub_index_spaces()) {
            auto named_is = str_subis.second;

            IndexVector indices;

            for(const auto& idx : named_is) {
                // find position in the root index space
                size_t pos = is.find_pos(idx);
                // named subspace should always find it
                EXPECTS(pos >= 0);

                size_t tile_idx     = 0;
                const auto& offsets = tiled_info_->tile_offsets_;
                // find in which tiles in the root it would be
                for(Index i = 0; pos >= offsets[i]; i++) { tile_idx = i; }
                indices.push_back(tile_idx);
            }
            // remove duplicates from the indices
            std::sort(indices.begin(), indices.end());
            auto last = std::unique(indices.begin(), indices.end());
            indices.erase(last, indices.end());

            IndexVector new_offsets;

            new_offsets.push_back(0);
            for(const auto& idx : indices) {
                new_offsets.push_back(new_offsets.back() +
                                      root_tiled_info_.lock()->tile_size(idx));
            }
            TiledIndexSpace tempTIS{};
            tempTIS.set_tiled_info(std::make_shared<TiledIndexSpaceInfo>(
              (*root_tiled_info_.lock()), named_is, new_offsets, indices));
            tempTIS.set_root(tiled_info_);
            tempTIS.set_parent(std::make_shared<TiledIndexSpace>(*this));
            tempTIS.compute_hash();

            tiled_info_->tiled_named_subspaces_.insert(
              {str_subis.first, tempTIS});
        }
    }

    template<std::size_t... Is>
    auto labels_impl(std::string id, Label start,
                     std::index_sequence<Is...>) const;

}; // class TiledIndexSpace

// Comparison operator implementations
inline bool operator==(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return lhs.is_identical(rhs);
}

inline bool operator<(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return lhs.is_less_than(rhs);
}

inline bool operator!=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (rhs < lhs);
}

inline bool operator<=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
    return (rhs <= lhs);
}

class TileLabelElement {
public:
    TileLabelElement()                        = default;
    TileLabelElement(const TileLabelElement&) = default;
    TileLabelElement(TileLabelElement&&)      = default;
    ~TileLabelElement()                       = default;
    TileLabelElement& operator=(const TileLabelElement&) = default;
    TileLabelElement& operator=(TileLabelElement&&) = default;

    TileLabelElement(const TiledIndexSpace& tis, Label label = 0) :
      tis_{tis},
      label_{label} {}

    const TiledIndexSpace& tiled_index_space() const { return tis_; }

    Label label() const { return label_; }

    bool is_compatible_with(const TiledIndexSpace& tis) const {
        return tis_.is_compatible_with(tis);
    }

private:
    TiledIndexSpace tis_;
    Label label_;
}; // class TileLabelElement

// Comparison operator implementations
inline bool operator==(const TileLabelElement& lhs,
                       const TileLabelElement& rhs) {
    return lhs.tiled_index_space() == rhs.tiled_index_space() &&
           lhs.label() == rhs.label();
}

inline bool operator<(const TileLabelElement& lhs,
                      const TileLabelElement& rhs) {
    return lhs.tiled_index_space() < rhs.tiled_index_space() ||
           (lhs.tiled_index_space() == rhs.tiled_index_space() &&
            lhs.label() < rhs.label());
}

inline bool operator!=(const TileLabelElement& lhs,
                       const TileLabelElement& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TileLabelElement& lhs,
                      const TileLabelElement& rhs) {
    return (rhs < lhs);
}

inline bool operator<=(const TileLabelElement& lhs,
                       const TileLabelElement& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TileLabelElement& lhs,
                       const TileLabelElement& rhs) {
    return (rhs <= lhs);
}

/**
 * @brief Index label to index into tensors. The labels used by the user need to
 * be positive.
 *
 */
class TiledIndexLabel {
public:
    // Constructor
    TiledIndexLabel()                       = default;
    TiledIndexLabel(const TiledIndexLabel&) = default;
    TiledIndexLabel(TiledIndexLabel&&)      = default;
    ~TiledIndexLabel()                      = default;

    TiledIndexLabel& operator=(const TiledIndexLabel&) = default;
    TiledIndexLabel& operator=(TiledIndexLabel&&) = default;

    /**
     * @brief Construct a new TiledIndexLabel object from a reference
     * TileLabelElement object and a label
     *
     * @param [in] t_is reference TileLabelElement object
     * @param [in] lbl input label (default: 0, negative values used internally)
     * @param [in] dep_labels set of dependent TiledIndexLabels (default: empty
     * set)
     */
    TiledIndexLabel(
      const TiledIndexSpace& tis, Label lbl = 0,
      const std::vector<TileLabelElement>& secondary_labels = {}) :
      TiledIndexLabel{TileLabelElement{tis, lbl}, secondary_labels} {
        // no-op
    }

    TiledIndexLabel(const TiledIndexSpace& tis, Label lbl,
                    const std::vector<TiledIndexLabel>& secondary_labels) :
      TiledIndexLabel{TileLabelElement{tis, lbl}, secondary_labels} {
        // no-op
    }

    TiledIndexLabel(const TiledIndexSpace& tis,
                    const std::vector<TiledIndexLabel>& secondary_labels) :
      TiledIndexLabel{TileLabelElement{tis, Label{0}}, secondary_labels} {
        // no-op
    }

    TiledIndexLabel(
      const TileLabelElement& primary_label,
      const std::vector<TileLabelElement>& secondary_labels = {}) :
      primary_label_{primary_label},
      secondary_labels_{secondary_labels} {
        validate();
    }

    TiledIndexLabel(const TileLabelElement& primary_label,
                    const std::vector<TiledIndexLabel>& secondary_labels = {}) :
      primary_label_{primary_label} {
        for(const auto& lbl : secondary_labels) {
            secondary_labels_.push_back(lbl.primary_label());
        }
        validate();
    }

    /**
     * @brief Construct a new TiledIndexLabel object from another one with input
     * dependent labels
     *
     * @param [in] t_il reference TiledIndexLabel object
     * @param [in] dep_labels set of dependent TiledIndexLabels
     */
    TiledIndexLabel(const TiledIndexLabel& til,
                    const std::vector<TileLabelElement>& secondary_labels) :
      primary_label_{til.primary_label()},
      secondary_labels_{secondary_labels} {
        validate();
    }

    TiledIndexLabel(const TiledIndexLabel& til,
                    const std::vector<TiledIndexLabel>& secondary_labels) :
      primary_label_{til.primary_label()} {
        for(const auto& lbl : secondary_labels) {
            secondary_labels_.push_back(lbl.primary_label());
        }
        validate();
    }

    /**
     * @brief Operator overload for () to construct dependent TiledIndexLabel
     * objects from the input TiledIndexLabels
     *
     * @returns a new TiledIndexLabel from this.
     */
    const TiledIndexLabel& operator()() const { return (*this); }

    /**
     * @brief Operator overload for () to construct dependent TiledIndexLabel
     * objects from the input TiledIndexLabels
     *
     * @tparam Args variadic template for multiple TiledIndexLabel object
     * @param [in] il1 input TiledIndexLabel object
     * @param [in] rest variadic template for rest of the arguments
     * @returns a new TiledIndexLabel object with corresponding dependent
     * TiledIndexLabels
     */
    template<typename... Args>
    TiledIndexLabel operator()(const TiledIndexLabel& il1, Args... rest) {
        EXPECTS(this->tiled_index_space().is_dependent());
        std::vector<TileLabelElement> secondary_labels;
        unpack(secondary_labels, il1, rest...);
        return {*this, secondary_labels};
    }

    /**
     * @brief Operator overload for () to construct dependent TiledIndexLabel
     * objects from the input TiledIndexLabels
     *
     * @param [in] dep_ilv
     * @returns
     */
    TiledIndexLabel operator()(
      const std::vector<TileLabelElement>& secondary_labels) {
        EXPECTS(this->tiled_index_space().is_dependent());
        return {*this, secondary_labels};
    }

    Label label() const { return primary_label_.label(); }

    /// @todo: this is never called from outside currently, should this be
    /// private and used internally?
    bool is_compatible_with(const TiledIndexSpace& tis) const {
        return tiled_index_space().is_compatible_with(tis);
    }

    /**
     * @brief Accessor method to dependent labels
     *
     * @returns a set of dependent TiledIndexLabels
     */
    const std::vector<TileLabelElement>& secondary_labels() const {
        return secondary_labels_;
    }

    /**
     * @brief The primary label pair that is composed of the reference
     * TiledIndexLabel and the label value
     *
     * @returns a pair of TiledIndexSpace object and Label value for the
     * TiledIndexLabel
     */
    const TileLabelElement& primary_label() const { return primary_label_; }

    /**
     * @brief Accessor to the reference TiledIndexSpace
     *
     * @returns the reference TiledIndexSpace object
     */
    const TiledIndexSpace& tiled_index_space() const {
        return primary_label_.tiled_index_space();
    }

    // Comparison operators
    friend bool operator==(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator<(const TiledIndexLabel& lhs,
                          const TiledIndexLabel& rhs);
    friend bool operator!=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator>(const TiledIndexLabel& lhs,
                          const TiledIndexLabel& rhs);
    friend bool operator<=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);
    friend bool operator>=(const TiledIndexLabel& lhs,
                           const TiledIndexLabel& rhs);

protected:
    // TiledIndexSpace* tis_;
    // Label label_;
    TileLabelElement primary_label_;
    std::vector<TileLabelElement> secondary_labels_;
    // std::vector<TiledIndexLabel> dep_labels_;

    /**
     * @brief Validates a TiledIndexLabel object with regard to its reference
     * TiledIndexSpace and dependent labels
     *
     */
    void validate() {
        // const auto& tis = tiled_index_space();
        // if(tis.is_dependent()) {
        //     const auto& sec_tis = tis.index_space().key_tiled_index_spaces();
        //     auto num_sec_tis    = sec_tis.size();
        //     auto num_sec_lbl    = secondary_labels_.size();
        //     EXPECTS((num_sec_lbl == 0) || (num_sec_lbl == num_sec_tis));
        //     for(size_t i = 0; i < num_sec_lbl; i++) {
        //         EXPECTS(secondary_labels_[i].is_compatible_with(sec_tis[i]));
        //     }
        // } else {
        //     EXPECTS(secondary_labels_.empty());
        // }
    }

private:
    void unpack(std::vector<TileLabelElement>& in_vec) {}

    template<typename... Args>
    void unpack(std::vector<TileLabelElement>& in_vec,
                const TileLabelElement& il1) {
        in_vec.push_back(il1);
    }

    template<typename... Args>
    void unpack(std::vector<TileLabelElement>& in_vec,
                const TiledIndexLabel& il1) {
        in_vec.push_back(il1.primary_label());
    }

    template<typename... Args>
    void unpack(std::vector<TileLabelElement>& in_vec,
                const TileLabelElement& il1, Args... rest) {
        in_vec.push_back(il1);
        unpack(in_vec, std::forward<Args>(rest)...);
    }

    template<typename... Args>
    void unpack(std::vector<TileLabelElement>& in_vec,
                const TiledIndexLabel& il1, Args... rest) {
        in_vec.push_back(il1.primary_label());
        unpack(in_vec, std::forward<Args>(rest)...);
    }
}; // class TiledIndexLabel

// Comparison operator implementations
inline bool operator==(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return lhs.primary_label() == rhs.primary_label() &&
           lhs.secondary_labels() == rhs.secondary_labels();
}

inline bool operator<(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (lhs.primary_label() < rhs.primary_label()) ||
           (lhs.primary_label() == rhs.primary_label() &&
            std::lexicographical_compare(
              lhs.secondary_labels().begin(), lhs.secondary_labels().end(),
              rhs.secondary_labels().begin(), rhs.secondary_labels().end()));
}

inline bool operator!=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (rhs < lhs);
}

inline bool operator<=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>=(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
    return (rhs <= lhs);
}

///////////////////////////////////////////////////////////

inline TiledIndexLabel TiledIndexSpace::label(std::string id, Label lbl) const {
    if(id == "all") return TiledIndexLabel{*this, lbl};
    return TiledIndexLabel{(*this)(id), lbl};
}

template<std::size_t... Is>
auto TiledIndexSpace::labels_impl(std::string id, Label start,
                                  std::index_sequence<Is...>) const {
    return std::make_tuple(label(id, start + Is)...);
}

} // namespace tamm

#endif // TAMM_TILED_INDEX_SPACE_HPP_
