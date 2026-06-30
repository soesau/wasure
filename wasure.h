// Copyright (c) 2026 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Sven Oesau
//

#ifndef CGAL_WATERTIGHT_SURFACE_RECONSTRUCTION_H
#define CGAL_WATERTIGHT_SURFACE_RECONSTRUCTION_H

//#define CGAL_SCANORIENT_DUMP_RANDOM_SCANLINES
//#define CGAL_SCANLINE_ORIENT_VERBOSE

#include "scanline_orient_normals.h"
#include <CGAL/Splitters.h>
#include <CGAL/Default_diagonalize_traits.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/boost/graph/alpha_expansion_graphcut.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/jet_estimate_normals.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for_each.h>
#endif

#include <unordered_map>
#include <vector>

namespace CGAL {

template<typename PointSet>
class Wasure {
public:
  using Point_set = PointSet;
  using Geom_traits = typename Kernel_traits<typename Point_set::Point_3>::Kernel;
  using FT = typename Geom_traits::FT;
  using Point_3 = typename Point_set::Point_3;

private:
  using Vector_3 = typename Point_set::Vector_3;
  using Segment_3 = typename Geom_traits::Segment_3;

  using Point_map = typename Point_set::Point_map;
  using Normal_map = typename Point_set::Vector_map;
  using Vector_map = typename Point_set::Vector_map;
  using Scan_angle_map = typename Point_set::template Property_map<float>;
  using Scan_ID_map = typename Point_set::template Property_map<unsigned char>;
  using Point_source_ID_map = typename Point_set::template Property_map<unsigned short>;

  using Search_traits_base = Search_traits_3<Geom_traits>;
  using Search_traits = Search_traits_adapter<typename Point_set::Index, Point_map, Search_traits_base>;
  using Distance = CGAL::Distance_adapter<typename Point_set::Index, typename Point_set::Point_map, CGAL::Euclidean_distance<Search_traits_base> >;
  using Splitter = CGAL::Sliding_midpoint<Search_traits>;
  using Search_tree = CGAL::Kd_tree<Search_traits, Splitter, CGAL::Tag_true, CGAL::Tag_true>;
  using Neighbor_search = CGAL::Orthogonal_k_neighbor_search<Search_traits, Distance, Splitter, Search_tree>;
  using Tree = typename Neighbor_search::Tree;

  using DT = CGAL::Delaunay_triangulation_3<Geom_traits>;
  using Cell_handle = typename DT::Cell_handle;
  using Vertex_handle = typename DT::Vertex_handle;
  using Facet = typename DT::Facet;
  using Locate_type = typename DT::Locate_type;

  using Point_generator = CGAL::Random_points_in_tetrahedron_3<Point_3>;

  Point_set &m_points;
  Point_map m_point_map;
  Normal_map m_normal_map;
  Vector_map m_los;
  Bbox_3 m_bbox;
  std::vector<std::uint32_t> m_simplified;
  std::vector<std::array<Vector_3, 3>> m_eigen_vectors;
  std::vector<std::array<double, 3>> m_eigen_values;

  Distance m_dist;
  std::shared_ptr<Tree> m_tree_ptr;
  DT m_triangulation;
  std::vector<Cell_handle> m_tets;
  std::unordered_map<Cell_handle, std::size_t> m_tet_indices;
  std::vector<std::pair<std::size_t, std::size_t>> m_edges;
  std::vector<double> m_edge_weights;
  std::vector<std::array<double, 3>> m_mass_function; // empty, occupied, unknown
  std::vector<double> m_tet_volumes;

  bool m_has_los;

public:
  template<typename Concurrency_tag = Sequential_tag>
  Wasure(Point_set&points)
    : m_points(points), m_point_map(points.point_map()), m_dist(points.point_map()),
      m_bbox(CGAL::bbox_3(CGAL::make_transform_iterator_from_property_map(m_points.begin(), m_point_map), CGAL::make_transform_iterator_from_property_map(m_points.end(), m_point_map)))
  {
    points.collect_garbage();
    for (auto idx : m_points) {
      Point_3 &p = get(m_point_map, idx);
      p = Point_3(p.x() - m_bbox.xmin(), p.y() - m_bbox.ymin(), p.z() - m_bbox.zmin());
    }
    m_tree_ptr.reset(new Tree(m_points.begin(), m_points.end(), Tree::Splitter(), Search_traits(m_points.point_map())));
    CGAL_assertion(!points.has_garbage());
    if (!points.has_normal_map()) {
      std::optional<Scan_angle_map> scan_angle_map = m_points.property_map<float>("scan_angle");
      std::optional<Scan_ID_map> scan_id_map = m_points.property_map<unsigned char>("scan_direction_flag");
      std::optional<Point_source_ID_map> point_source_id_map = m_points.property_map<unsigned short>("point_source_ID");
      //std::optional<
      CGAL_assertion(scan_angle_map.has_value() && (scan_id_map.has_value() || point_source_id_map.has_value()));
      m_los = m_points.add_property_map<Vector_3>("los").first;
      m_has_los = true;
      //CGAL::jet_estimate_normals<Concurrency_tag>(m_points, 100, CGAL::parameters::point_map(m_point_map).normal_map(m_normal_map));
      if (scan_id_map.has_value())
        CGAL::scanline_orient_normals(m_points,
          CGAL::parameters::point_map(m_points.point_map()).
          normal_map(m_los).
          scan_angle_map(scan_angle_map.value()).
          scanline_id_map(scan_id_map.value()));
      else
        CGAL::scanline_orient_normals(m_points,
          CGAL::parameters::point_map(m_points.point_map()).
          normal_map(m_los).
          scan_angle_map(scan_angle_map.value()).
          scanline_id_map(point_source_id_map.value()));

      for (auto idx : m_points) {
        Vector_3 &v = get(m_los, idx);
        v = Vector_3(v.x() * 200.0, v.y() * 200.0, v.z() * 200.0);
      }

      dump_los("los.ply");
    }
    else m_has_los = false;
  }

  ~Wasure() {}

  template<typename Concurrency_tag = Sequential_tag, typename DiagonalizeTraits = CGAL::Default_diagonalize_traits<double, 3>>
  void compute_features() {
    std::vector<Point_3> pts;
    pts.reserve(150);

    m_eigen_values.resize(m_points.size());
    m_eigen_vectors.resize(m_points.size());

    for (auto idx : m_points) {
      double entropy = 1000000000;
      pts.clear();
      std::array<FT, 3> sum = { 0, 0, 0 };
      Neighbor_search search(*m_tree_ptr, get(m_point_map, idx), 150, 0, true, m_dist);
      for (typename Neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
        pts.push_back(get(m_point_map, it->first));

        sum[0] += pts.back().x();
        sum[1] += pts.back().y();
        sum[2] += pts.back().z();

        if (pts.size() < 10)
          continue;

        if (it->second < 0.02 && pts.size() < 30)
          continue;

        std::array<FT, 3> eigen_values;
        std::array<Vector_3, 3> eigen_vectors;

        pca<DiagonalizeTraits>(pts, Point_3(sum[0] / pts.size(), sum[1] / pts.size(), sum[2] / pts.size()), eigen_values, eigen_vectors);

        double cur_entropy = 0;

        std::array<FT, 3> ev = { .0, .0, .0 };

        // Normalization like in reference implementation
        double sum = eigen_values[0] + eigen_values[1] + eigen_values[2];
        ev[0] = (eigen_values[0] - eigen_values[1]) / sum;
        ev[1] = (eigen_values[1] - eigen_values[2]) / sum;
        ev[2] = eigen_values[2] / sum;

        for (int i = 0; i < 3; i++)
          cur_entropy += -ev[0] * log(ev[0]);

        if (cur_entropy < entropy) {
          entropy = cur_entropy;
          m_eigen_values[idx] = eigen_values;
          m_eigen_vectors[idx] = eigen_vectors;
        }
      }
      assert(entropy != 1000000000);
    }

    orient_features();

    Normal_map normal_map = m_points.normal_map();
    for (auto idx : m_points) {
      Point_3 p = get(m_point_map, idx);
      Vector_3 n = get(normal_map, idx);
      Vector_3 np = Vector_3(p.x(), p.y(), p.z());
      assert(n * np > 0);
      Vector_3 ev = m_eigen_vectors[idx][2];
      assert(n * ev > 0);
      assert(np * ev > 0);
    }

    dump_features("out_sig.ply", true);
  }

  template<typename Concurrency_tag = Sequential_tag, typename DiagonalizeTraits = CGAL::Default_diagonalize_traits<double, 3>>
  void compute_features_svd() {
    std::vector<Point_3> pts;
    pts.reserve(150);
    Eigen::MatrixXf mat(150, 3);

    m_eigen_values.resize(m_points.size());
    m_eigen_vectors.resize(m_points.size());

    for (auto idx : m_points) {
      double entropy = 1000000000;
      pts.clear();
      std::array<float, 3> sum = { 0, 0, 0 };
      Neighbor_search search(*m_tree_ptr, get(m_point_map, idx), 150, 0, true, m_dist);
      int k = 0;
      for (typename Neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
        pts.push_back(get(m_point_map, it->first));
        const Point_3 &p = get(m_point_map, it->first);
        mat.row(k) << p.x(), p.y(), p.z();
        k++;

        sum[0] += pts.back().x();
        sum[1] += pts.back().y();
        sum[2] += pts.back().z();

        //if (k < 10)
        //  continue;

        if (it->second < 0.02 && k < 30)
          continue;

        std::array<FT, 3> singular_values;
        std::array<Vector_3, 3> eigen_vectors;

        svd(mat, sum, singular_values, eigen_vectors);

        double cur_entropy = 0;

        std::array<FT, 3> ev = { .0, .0, .0 };

        // Normalization like in reference implementation
        double sum = singular_values[0] + singular_values[1] + singular_values[2];
        ev[0] = (singular_values[0] - singular_values[1]) / sum;
        ev[1] = (singular_values[1] - singular_values[2]) / sum;
        ev[2] = singular_values[2] / sum;

        for (int i = 0; i < 3; i++)
          cur_entropy += -ev[0] * log(ev[0]);

        if (cur_entropy < entropy) {
          entropy = cur_entropy;
          m_eigen_values[idx] = singular_values;
          m_eigen_vectors[idx] = eigen_vectors;
        }
      }
      assert(entropy != 1000000000);
    }

    orient_features();

    Normal_map normal_map = m_points.normal_map();
    for (auto idx : m_points) {
      Point_3 p = get(m_point_map, idx);
      Vector_3 n = get(normal_map, idx);
      Vector_3 np = Vector_3(p.x(), p.y(), p.z());
      assert(n * np > 0);
      Vector_3 ev = m_eigen_vectors[idx][2];
      assert(n * ev > 0);
      assert(np * ev > 0);
    }

    dump_features("out_sig_svd.ply", true);
  }

  void compute_mass_function(std::uint32_t num_samples = 40) {
    if (m_triangulation.number_of_vertices() == 0)
      m_triangulation.insert(CGAL::make_transform_iterator_from_property_map(m_points.begin(), m_point_map), CGAL::make_transform_iterator_from_property_map(m_points.end(), m_point_map));

    m_tets.clear();
    m_tets.reserve(m_triangulation.number_of_cells() + 1);
    m_tet_indices[m_triangulation.infinite_cell()] = m_tets.size() - 1;
    for (auto it = m_triangulation.cells_begin(); it != m_triangulation.cells_end(); ++it) {
      m_tets.push_back(it);
      m_tet_indices[it] = m_tets.size() - 1;
    }

    m_edges.clear();
    m_edges.reserve(m_triangulation.number_of_facets());
    m_edge_weights.clear();
    m_edge_weights.reserve(m_triangulation.number_of_facets());

    double edge_weight_sum = 0;
    double tet_weight_sum = 0;

    for (auto it = m_triangulation.finite_facets_begin(); it != m_triangulation.finite_facets_end(); ++it) {
      Cell_handle c1 = it->first;
      Cell_handle c2 = m_triangulation.mirror_facet(*it).first;
      std::size_t idx1 = m_tet_indices[c1];
      std::size_t idx2 = m_tet_indices[c2];
      m_edges.push_back(std::make_pair(idx1, idx2));
      m_edge_weights.push_back(CGAL::sqrt(m_triangulation.triangle(*it).squared_area()));
      edge_weight_sum += m_edge_weights.back();
    }

    std::cout << "edge weight sum: " << edge_weight_sum << std::endl;

    m_mass_function.resize(m_tets.size(), std::array<double, 3>{0, 0, 0});
    m_tet_volumes.resize(m_tets.size(), 0);

    for (int id = 0; id < m_tets.size();id++) {
      if (m_triangulation.is_infinite(m_tets[id])) {
        m_mass_function[id] = { 1, 0, 0 };
        continue;
      }

      m_tet_volumes[id] = m_triangulation.tetrahedron(m_tets[id]).volume();
      tet_weight_sum += m_tet_volumes[id];

      compute_local_mass(m_tets[id], m_mass_function[id], num_samples);

      double sum = m_mass_function[id][0] + m_mass_function[id][1] + m_mass_function[id][2];
      if (sum < 0.99 || sum > 1.01)
        std::cout << sum << " is out of bounds" << std::endl;
    }

    std::cout << "tet weight sum: " << tet_weight_sum << std::endl;
    tet_weight_sum = edge_weight_sum / tet_weight_sum;

    for (int id = 0; id < m_tets.size(); id++) {
      if (m_triangulation.is_infinite(m_tets[id]))
        continue;

      m_tet_volumes[id] *= tet_weight_sum;
    }

    dump_mass_function("mf_after_local.ply");

    if (false && m_has_los) {
      compute_origin_mass();
      dump_mass_function("mf_after_origin.ply");
      penalize_flight_path();
      dump_mass_function("mf_after_flight.ply");
    }
  }

  template<typename PointOutputIterator, typename IndexOutputIterator>
  void extract_surface_from_weights(double isovalue, PointOutputIterator pit, IndexOutputIterator iit) {
    std::unordered_map<Vertex_handle, std::uint32_t> vh2idx;
    for (std::uint32_t i = 0; i < m_tets.size(); ++i) {
      if (m_triangulation.is_infinite(m_tets[i]))
        continue;

      double e1 = m_mass_function[i][0];
      int i1 = (e1 < isovalue) ? -1 : 1;

      for (int j = 0; j < 4; j++) {
        Cell_handle ch = m_triangulation.mirror_facet(Facet(m_tets[i], j)).first;
        double e2 = m_mass_function[m_tet_indices[ch]][0];
        int i2 = (e2 < isovalue) ? -1 : 1;
        if (i1 * i2 < 0) {
          std::array<Vertex_handle, 3> vhs = m_triangulation.vertices(Facet(m_tets[i], j));
          std::vector<std::uint32_t> indices(3);
          for (std::uint32_t k = 0; k < 3; k++) {
            auto res = vh2idx.insert(std::make_pair(vhs[k], vh2idx.size()));
            indices[k] = res.first->second;
            if (res.second)
              *pit++ = vhs[k]->point();
          }

          *iit++ = std::move(indices);
        }
      }
    }
  }

  template<typename PointOutputIterator, typename IndexOutputIterator>
  void extract_surface(FT lambda, PointOutputIterator pit, IndexOutputIterator iit) {
    std::vector<std::size_t> labels(m_tets.size());
    std::vector<double> m_edge_weights_lambda(m_edge_weights.size());
    std::vector<double> label_values = { 0, 1 };
    for (std::size_t i = 0;i<m_edge_weights.size();i++)
      m_edge_weights_lambda[i] = lambda * m_edge_weights[i];

    std::vector<std::vector<double>> m_cost_matrix;
    m_cost_matrix.resize(label_values.size());
    for (std::size_t l = 0;l<label_values.size();l++)
      m_cost_matrix[l].resize(m_tets.size());

    for (int id = 0; id < m_tets.size(); id++) {
      if (m_triangulation.is_infinite(m_tets[id])) {
        for (std::size_t l = 0; l < label_values.size(); l++)
          m_cost_matrix[l][id] = (l == 0) ? 0 : 1000;
        continue;
      }

      for (std::size_t l = 0; l < label_values.size(); l++)
        m_cost_matrix[l][id] = (std::max)(0.0, (1 - m_mass_function[id][l]) * m_tet_volumes[id] * (1 - lambda));
    }

    CGAL::alpha_expansion_graphcut(m_edges, m_edge_weights_lambda, m_cost_matrix, labels);

    int cnt = 0;
    for (std::size_t i : labels)
      if (i == 0)
        cnt++;

    std::cout << cnt << " tets labeled as empty, " << m_tets.size() - cnt << " tets labeled as occupied" << std::endl;

    std::unordered_map<Vertex_handle, std::uint32_t> vh2idx;
    for (std::uint32_t i = 0; i < m_tets.size(); ++i) {
      if (m_triangulation.is_infinite(m_tets[i]) || labels[i] == 0)
        continue;

      for (int j = 0;j<4;j++) {
        auto mirror_facet = m_triangulation.mirror_facet(Facet(m_tets[i], j));
        Cell_handle ch = mirror_facet.first;
        if (labels[m_tet_indices[ch]] == 0) {
          std::array<Vertex_handle, 3> vhs = m_triangulation.vertices(mirror_facet);
          std::vector<std::uint32_t> indices(3);
          for (std::uint32_t k = 0;k<3;k++) {
            auto res = vh2idx.insert(std::make_pair(vhs[k], vh2idx.size()));
            indices[k] = res.first->second;
            if (res.second)
              *pit++ = vhs[k]->point();
          }

          *iit++ = std::move(indices);
        }
      }
    }
  }

  std::size_t adaptive_triangulation(FT pscale, int iterations = 10) {
    std::vector<Point_3> points;
    create_initial_triangulation(pscale);
    vertex_k_means(points, iterations);
    return m_triangulation.number_of_vertices();
    //m_triangulation.clear();
    //m_triangulation.insert(points.begin(), points.end());
  }

  void simplify(FT density) {
    CGAL_assertion(m_eigen_values.size() != m_points.size());
    if (m_eigen_values.size() != m_points.size())
      return;

    std::vector<bool> skip(m_points.size(), false);
    m_simplified.clear();
    m_simplified.reserve(m_points.size());

    for (auto idx : m_points) {
      if (skip[idx])
        continue;
      m_simplified.push_back(idx);
      const Point_3 &p = get(m_point_map, idx);
      Neighbor_search search(*m_tree_ptr, p, 50, 0, true, m_dist);
      for (typename Neighbor_search::iterator it = search.begin() + 1; it != search.end(); ++it) {
        Vector_3 d = get(m_point_map, it->first) - p;

        for (int i = 0; i < 3; i++)
          if (CGAL::abs(d * m_eigen_vectors[idx][i]) > density * m_eigen_values[idx][i]) {
            skip[it->first] = true;
            break;
          }
      }
    }
  }

private:
  template<typename DiagonalizeTraits>
  void pca(const std::vector<Point_3> &pts, const Point_3 &mean, std::array<double, 3> &eigen_values, std::array<Vector_3, 3> &eigen_vectors) {
    std::array<double, 6> covariance = make_array(.0, .0, .0, .0, .0, .0);
    for (const Point_3 &p : pts) {
      Vector_3 d = p - mean;
      covariance[0] += d.x() * d.x();
      covariance[1] += d.x() * d.y();
      covariance[2] += d.x() * d.z();
      covariance[3] += d.y() * d.y();
      covariance[4] += d.y() * d.z();
      covariance[5] += d.z() * d.z();
    }

    std::array<double, 9> evectors = make_array(.0, .0, .0, .0, .0, .0, .0, .0, .0);

    DiagonalizeTraits::diagonalize_selfadjoint_covariance_matrix(covariance, eigen_values, evectors);

    double tmp = eigen_values[0];
    eigen_values[0] = eigen_values[2];
    eigen_values[2] = tmp;

    for (int i = 0; i < 3; i++)
      if (eigen_values[i] < 0.0000001 || eigen_values[i] != eigen_values[i])
        eigen_values[i] = 0.0000001;

    eigen_vectors[0] = Vector_3(evectors[6], evectors[7], evectors[8]);
    eigen_vectors[1] = Vector_3(evectors[3], evectors[4], evectors[5]);
    eigen_vectors[2] = Vector_3(evectors[0], evectors[1], evectors[2]);
  }

  void svd(Eigen::MatrixXf &mat, const std::array<float, 3> &sum, std::array<double, 3> &sv, std::array<Vector_3, 3> &ev) {
    Eigen::MatrixXf m = mat.rowwise() - mat.colwise().mean();
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf svm = svd.singularValues();
    Eigen::MatrixXf evm = svd.matrixV();

    ev[0] = Vector_3(evm(0, 0), evm(0, 1), evm(0, 2));
    ev[1] = Vector_3(evm(1, 0), evm(1, 1), evm(1, 2));
    ev[2] = Vector_3(evm(2, 0), evm(2, 1), evm(2, 2));

    double v_min = 0.0000001;
    sv[0] = (svm(0) < v_min || std::isnan(svm(0))) ? v_min : svm(0);
    sv[1] = (svm(1) < v_min || std::isnan(svm(1))) ? v_min : svm(1);
    sv[2] = (svm(2) < v_min || std::isnan(svm(2))) ? v_min : svm(2);
  }

  void vertex_k_means(std::vector<Point_3>& points, std::size_t iterations) {
    points.clear();

    CGAL::Random random = CGAL::get_default_random();
    CGAL::Unique_hash_map<Vertex_handle, std::size_t> vh2idx;
    std::vector<std::array<FT, 3>> pos, normal, scale, los;
    std::vector<std::size_t> count;
    pos.resize(m_triangulation.number_of_vertices());
    normal.resize(m_triangulation.number_of_vertices());
    scale.resize(m_triangulation.number_of_vertices());
    los.resize(m_triangulation.number_of_vertices());
    count.resize(m_triangulation.number_of_vertices());

    bool has_los = m_points.has_property_map<Vector_3>("los");

    std::size_t idx = 0;
    for (auto it : m_triangulation.finite_vertex_handles())
      vh2idx[it] = idx++;

    for (std::size_t i = 0; i < iterations; i++) {
      std::fill(pos.begin(), pos.end(), std::array<FT, 3>({ .0, .0, .0 }));
      std::fill(normal.begin(), normal.end(), std::array<FT, 3>({ .0, .0, .0 }));
      std::fill(scale.begin(), scale.end(), std::array<FT, 3>({ .0, .0, .0 }));
      std::fill(los.begin(), los.end(), std::array<FT, 3>({ .0, .0, .0 }));
      std::fill(count.begin(), count.end(), 0);

      for (auto idx : m_points) {
        Point_3& p = get(m_point_map, idx);
        Vertex_handle vh = m_triangulation.nearest_vertex(p);
        std::size_t vhidx = vh2idx[vh];

        pos[vhidx][0] += p.x();
        pos[vhidx][1] += p.y();
        pos[vhidx][2] += p.z();
        count[vhidx]++;

        if (i == (iterations - 1)) {
          normal[vhidx][0] += m_eigen_vectors[idx][2].x();
          normal[vhidx][1] += m_eigen_vectors[idx][2].y();
          normal[vhidx][2] += m_eigen_vectors[idx][2].z();

          scale[vhidx][0] += m_eigen_values[idx][0];
          scale[vhidx][1] += m_eigen_values[idx][1];
          scale[vhidx][2] += m_eigen_values[idx][2];

          if (has_los) {
            Vector_3 vlos = get(m_los, idx);
            los[vhidx][0] += vlos.x();
            los[vhidx][1] += vlos.y();
            los[vhidx][2] += vlos.z();
          }
        }
      }

      for (auto it : m_triangulation.finite_vertex_handles()) {
        std::size_t vhidx = vh2idx[it];
        std::array<FT, 3>& mean = pos[vhidx];
        std::size_t c = count[vhidx];
        if (c == 0)
          continue;

        for (FT& x : mean) x /= c;

        if (i < (iterations - 1))
          m_triangulation.move(it, Point_3(mean[0], mean[1], mean[2]));
        else {
          points.push_back(Point_3(mean[0], mean[1], mean[2]));
          if (!has_los && random.uniform_01<double>() > 0.8) {
            std::array<FT, 3>& norm = normal[vhidx];
            std::array<FT, 3>& sca = scale[vhidx];
            for (FT& x : norm) x /= c;
            for (FT& x : sca) x /= c;

            double largest_ev = (std::max)({ sca[0], sca[1], sca[2] }) / 3.0;

            points.push_back(Point_3(mean[0] - largest_ev * norm[0], mean[1] - largest_ev * norm[1], mean[2] - largest_ev * norm[2]));
            //std::array<FT, 3>& lsum = los[vhidx];
            //it->info().line_of_sight = Vector_3(lsum[0] / c, lsum[1] / c, lsum[2] / c);
          }
        }
      }
    }
  }

  void create_initial_triangulation(FT pscale) {
    m_triangulation.clear();

    std::vector<std::size_t> indices(m_points.size());
    std::iota(indices.begin(), indices.end(), 0);
    CGAL::cpp98::random_shuffle(indices.begin(), indices.end(), CGAL::get_default_random());

    for (std::size_t idx : indices) {
      Locate_type locate_type;
      Point_3& p = get(m_point_map, idx);
      int li, lj;
      Cell_handle ch = m_triangulation.locate(p, locate_type, li, lj);
      if (locate_type > Locate_type::CELL)
        m_triangulation.insert(p);
      else {
        bool insert = true;
        for (int i = 0; i < 4; i++) {
          std::array<double, 3> coef = calculate_point_coefficients(p, ch->vertex(i)->point(), m_eigen_vectors[idx]);
          for (int d = 0; d < 3; d++)
            if (fabs(coef[d]) < m_eigen_values[idx][d] * pscale) {
              insert = false;
              i = 5; // break outer loop
              break;
            }
        }
        if (insert)
          m_triangulation.insert(p);
      }
    }
  }

  void compute_local_mass(Cell_handle ch, std::array<FT, 3> &mass, std::uint32_t nb_samples) {
    Point_generator pg(m_triangulation.tetrahedron(ch));
    for (std::uint32_t i = 0; i < nb_samples; ++i) {
      Point_3 s;
      if (i == 0)
        s = CGAL::centroid(ch->vertex(0)->point(), ch->vertex(1)->point(), ch->vertex(2)->point(), ch->vertex(3)->point());
      else
        s = *pg++;

      // Compute local mass for each sample
      Neighbor_search search(*m_tree_ptr, s, 30, 0, true, m_dist);
      std::array<double, 3> sample_mass = { 0, 0, 1 };
      for (typename Neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
        Point_3 p = get(m_point_map, it->first);
        double largest_ev = m_eigen_values[it->first][0];
        std::array<double, 3> point_mass = { 0, 0, 0 };
        std::array<double, 3> point_coefficients = calculate_point_coefficients(p, s, m_eigen_vectors[it->first]);
        compute_dst_local_sample(point_coefficients, m_eigen_values[it->first], 1.0, largest_ev, largest_ev, point_mass);
        ds_score(sample_mass, point_mass, sample_mass);
        regularize(sample_mass);
      }
      mass[0] += sample_mass[0];
      mass[1] += sample_mass[1];
      mass[2] += sample_mass[2];
    }

    CGAL_assertion(!std::isnan(mass[0]));
    CGAL_assertion(!std::isnan(mass[1]));
    CGAL_assertion(!std::isnan(mass[2]));

    mass[0] /= nb_samples;
    mass[1] /= nb_samples;
    mass[2] /= nb_samples;
  }

  void compute_origin_mass(std::uint32_t nb_samples = 40, unsigned int ratio = 0) {
    std::optional<Vector_map> res = m_points.property_map<Vector_3>("los");
    CGAL_assertion(res.has_value());
    Vector_map los = res.value();
    Cell_handle hint = Cell_handle();
    for (int i = 0;i<m_points.size();i++) {
//       if (ratio != 0 && i % ratio != 0)
//         continue;

      Point_3 &p = get(m_point_map, i);
      Vector_3 &vlos = get(los, i);

      if (std::isnan(vlos.x()) || std::isnan(vlos.y()) || std::isnan(vlos.z()))
        continue;

      Point_3 center = p + vlos;
      Point_3 mirrored_center = p - vlos;

      walk_los(p, center, mirrored_center, hint, i, nb_samples);
    }
  }

  double l(double v, double mi, double ma) {
    v -= mi;
    return (std::min)(v, ma) / ma;
  }

  void dump_features(const std::string& filename, bool normalize) {
    double mi = 1000, ma = -mi;
    if (normalize)
      for (int i = 0; i < m_points.size(); i++) {
        if (m_eigen_values[i][0] != m_eigen_values[i][0] || m_eigen_values[i][1] != m_eigen_values[i][1] || m_eigen_values[i][2] != m_eigen_values[i][2])
          continue;
        for (int j = 0; j < 3; j++) {
          mi = (std::min)(mi, m_eigen_values[i][j]);
          ma = (std::max)(ma, m_eigen_values[i][j]);
        }
        assert(m_eigen_values[i][0] >= m_eigen_values[i][1]);
        assert(m_eigen_values[i][1] >= m_eigen_values[i][2]);
      }

    std::ofstream ofile(filename);
    ofile << std::setprecision(17);
    ofile << "ply" << std::endl <<
      "format ascii 1.0" << std::endl <<
      "element vertex " << m_points.size() << std::endl <<
      "comment " << mi << " " << ma << std::endl <<
      "property double x" << std::endl <<
      "property double y" << std::endl <<
      "property double z" << std::endl <<
      "property uchar red" << std::endl <<
      "property uchar green" << std::endl <<
      "property uchar blue" << std::endl <<
      "end_header" << std::endl;

    ma += mi;
    if (ma > 20)
      ma = 20;

    for (int i = 0;i<m_points.size();i++) {
      ofile << get(m_point_map, i);

      if (normalize)
        ofile << " " << int(l(m_eigen_values[i][0], mi, ma) * 255.0 + 0.5) << " " << int(l(m_eigen_values[i][1], mi, ma) * 255.0 + 0.5) << " " << int(l(m_eigen_values[i][2], mi, ma) * 255.0 + 0.5) << "\n";
      else
        ofile << " " << int(l(m_eigen_values[i][0], 0, 1.0) * 255.0 + 0.5) << " " << int(l(m_eigen_values[i][1], 0, 1.0) * 255.0 + 0.5) << " " << int(l(m_eigen_values[i][2], 0, 1.0) * 255.0 + 0.5) << "\n";
    }
  }

  void dump_mass_function(const std::string& filename) {
    std::ofstream ofile(filename);
    ofile << std::setprecision(17);
    ofile << "ply" << std::endl <<
      "format ascii 1.0" << std::endl <<
      "element vertex " << m_tet_indices.size() << std::endl <<
      "property double x" << std::endl <<
      "property double y" << std::endl <<
      "property double z" << std::endl <<
      "property uchar red" << std::endl <<
      "property uchar green" << std::endl <<
      "property uchar blue" << std::endl <<
      "end_header" << std::endl;

    for (auto ch = m_triangulation.cells_begin(); ch != m_triangulation.cells_end(); ++ch) {
      std::size_t chidx = m_tet_indices[ch];
      ofile << CGAL::centroid(ch->vertex(0)->point(), ch->vertex(1)->point(), ch->vertex(2)->point(), ch->vertex(3)->point());

      ofile << " " << int(m_mass_function[chidx][0] * 255.0 + 0.5) << " " << int(m_mass_function[chidx][1] * 255.0 + 0.5) << " " << int(m_mass_function[chidx][2] * 255.0 + 0.5) << "\n";
    }
  }

  void dump_los(const std::string& filename) {
    if (!m_has_los) {
      std::cout << "dump_los called without having line of sight data" << std::endl;
      return;
    }
    std::ofstream ofile(filename);
    ofile << std::setprecision(17);
    ofile << "ply" << std::endl <<
      "format ascii 1.0" << std::endl <<
      "element vertex " << m_points.size() << std::endl <<
      "property double x" << std::endl <<
      "property double y" << std::endl <<
      "property double z" << std::endl <<
      "property double nx" << std::endl <<
      "property double ny" << std::endl <<
      "property double nz" << std::endl <<
      "end_header" << std::endl;

    for (auto idx : m_points) {
      ofile << get(m_point_map, idx);
      Vector_3 los = get(m_los, idx);
      if (std::isnan(los.x()) || std::isnan(los.y()) || std::isnan(los.z()))
        ofile << " 0 0 0\n";
      else ofile << " " << los << "\n";
    }
  }

  void penalize_flight_path() {
    std::optional<Vector_map> res = m_points.property_map<Vector_3>("los");
    CGAL_assertion(res.has_value());
    Vector_map los = res.value();

    Cell_handle hint = Cell_handle();
    for (std::uint32_t i = 0; i < m_points.size(); ++i) {
      const Point_3 &p = get(m_point_map, i);
      const Vector_3 &vlos = get(los, i);

      if (std::isnan(vlos.x()) || std::isnan(vlos.y()) || std::isnan(vlos.z()))
        continue;

      Cell_handle ch = m_triangulation.locate(p + vlos, hint);
      hint = ch;

      int ind_inf;
      if (ch->has_vertex(m_triangulation.infinite_vertex(), ind_inf))
        continue;

      std::size_t cidx = m_tet_indices[ch];

      ds_score(m_mass_function[cidx], {0.05, 0, 0.95} , m_mass_function[cidx]);
    }
  }

  void orient_features() {
    Vector_map ref = (m_has_los) ? m_los : m_normal_map;
    for (auto idx : m_points) {
      Vector_3 normal = get(ref, idx);
      if (normal * m_eigen_vectors[idx][2] < 0)
        m_eigen_vectors[idx][2] = -m_eigen_vectors[idx][2];
    }
  }

  void compute_dst_local_sample(const std::array<double, 3> &pcoeff, const std::array<double, 3> &eigen_values, double coef_conf, double pdfs_e, double pdfs_o, std::array<double, 3> &mass) {
    double nscale = eigen_values[2];
    if (nscale <= 0.00001)
      nscale = 0.00001;

    // Is sample point in front of the surface point?
    if (pcoeff[2] > 0) {
      mass[0] = 1 - 0.5 * (exp(-fabs(pcoeff[2]) / nscale));
      mass[1] = 1 - mass[0];
    }
    else if (pcoeff[2] < 0) {
      mass[1] = 1 - 0.5 * (exp(-fabs(pcoeff[2]) / nscale));
      mass[0] = 1 - mass[1];
    }
    else mass[0] = mass[1] = 0.5;

    for (int i = 0;i<3;i++)
      if (eigen_values[i] <= 0)
        mass[0] = mass[1] = 0;
      else {
        const double score_pdf = exp(-(pcoeff[i] / eigen_values[i]) * (pcoeff[i] / eigen_values[i]));
        mass[0] *= score_pdf;
        mass[1] *= score_pdf;
      }

    mass[0] = mass[0] * exp(-(fabs(pcoeff[2]) / (pdfs_e)) * (fabs(pcoeff[2]) / (pdfs_e)));
    mass[1] = mass[1] * exp(-(fabs(pcoeff[2]) / (pdfs_o)) * (fabs(pcoeff[2]) / (pdfs_o)));
    mass[0] = mass[0] * coef_conf;
    mass[1] = mass[1] * coef_conf;

    regularize(mass);
  }

  void compute_dst_los_samples(const Cell_handle &ch, const Point_3 &p, const Point_3& center, std::size_t pidx, std::uint32_t nb_samples) {
    std::size_t cidx = m_tet_indices[ch];
    Point_generator pg(m_triangulation.tetrahedron(ch));
    for (std::uint32_t i = 0; i < nb_samples; ++i) {
      Point_3 s;
      if (i == 0)
        s = CGAL::centroid(ch->vertex(0)->point(), ch->vertex(1)->point(), ch->vertex(2)->point(), ch->vertex(3)->point());
      else
        s = *pg++;

      std::array<double, 3> point_coefficients = calculate_point_coefficients(p, s, m_eigen_vectors[cidx]);
      std::array<double, 3> &mass = m_mass_function[cidx]; // pe1, po1, pu1
      std::array<double, 3> mass2;

      //double angle;
      compute_dst_mass_beam(point_coefficients, m_eigen_values[cidx], /*angle, ANGLE_SCALE,*/ 1.0, mass2);
      mass[0] *= 1.0;
      mass[1] = 0;
      mass[2] = 1.0 - mass[0];
      ds_score(mass, mass2, mass);
      regularize(mass);
    }
  }

  void compute_dst_mass_beam(const std::array<double, 3>& pcoeff, const std::array<double, 3>& eigen_values, /*double angle, double angle_scale,*/ double coef_conf, std::array<double, 3> &mass)
  {
    double c3 = pcoeff[2];
    double nscale = eigen_values[2];
    if (nscale <= 0) nscale = 0.000001;
    if (c3 >= 0)
    {
      mass[0] = 1 - (exp(-fabs(c3) / (nscale * 2)));
      mass[1] = 0;
    }
    else if (c3 < 0)
      mass[0] = mass[1] = 0;

    mass[0] *= coef_conf;
    mass[1] *= coef_conf;
    regularize(mass);
  }

  std::array<double, 3> calculate_point_coefficients(const Point_3& s, const Point_3& p, const std::array<Vector_3, 3> &eigen_vectors) const {
    std::array<double, 3> coeffs;
    for (int i = 0; i < 3; i++)
      coeffs[i] = (p - s) * eigen_vectors[i] / (eigen_vectors[i] * eigen_vectors[i]);
    return coeffs;
  }

  void ds_score(const std::array<double, 3> &v1, const std::array<double, 3> &v2, std::array<double, 3> &v3) {
    double vK = v1[1] * v2[0] + v1[0] * v2[1];
    CGAL_assertion(vK != 1);
    v3[0] = (v1[0] * v2[0] + v1[0] * v2[2] + v1[2] * v2[0]) / (1 - vK);
    v3[1] = (v1[1] * v2[1] + v1[1] * v2[2] + v1[2] * v2[1]) / (1 - vK);
    v3[2] = (v1[2] * v2[2]) / (1 - vK);
  }

  void regularize(std::array<double, 3> &mass) const {
    mass[0] = std::clamp(mass[0], 0.0, 1.0);
    mass[1] = std::clamp(mass[1], 0.0, 1.0);
    mass[2] = 1 - mass[0] - mass[1];
    if (mass[2] < 0) {
      double sum = mass[0] + mass[1];
      mass[0] /= sum;
      mass[1] /= sum;
      mass[2] = 0;
    }
  }

  Cell_handle walk_los(const Point_3& p, const Point_3& center, const Point_3& mirrored, Cell_handle hint, std::size_t pidx, std::uint32_t nb_samples) {
    CGAL_assertion(m_triangulation.number_of_vertices() > 1);
    Segment_3 seg(center, p);

    Cell_handle start = m_triangulation.locate(center, hint);
    Cell_handle start_cell = start;

    int ind_inf;
    if (start->has_vertex(m_triangulation.infinite_vertex(), ind_inf))
      start = start->neighbor(ind_inf);

    Cell_handle cur = start, prev = Cell_handle();

    bool found_next = true;
    for (int iteration = 0; iteration < 10000 && found_next; iteration++) {
      found_next = false;
      Tetrahedron_3 tet = m_triangulation.tetrahedron(cur);

      if (CGAL::do_intersect(tet, seg)) {
        //compute_dst_los_samples(c, p, center, pidx, nb_samples);
      }

      std::vector<Point_3> pts = { cur->vertex(0)->point(), cur->vertex(1)->point(), cur->vertex(2)->point(), cur->vertex(3)->point() };
      for (int i = 0; i < 4; i++) {
        Cell_handle next = cur->neighbor(i);
        if (next == prev)
          continue;
        Point_3 bkp = pts[i];
        pts[i] = mirrored;
        if (CGAL::orientation(pts[0], pts[1], pts[2], pts[3]) != CGAL::NEGATIVE) {
          pts[i] = bkp;
          continue;
        }
        if (next->has_vertex(m_triangulation.infinite_vertex(), ind_inf))
          return start_cell;

        prev = cur;
        cur = next;
        found_next = true;
      }
    }

    return start_cell;
  }
};

} // namespace CGAL

#endif
