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

#include <CGAL/Splitters.h>
#include <CGAL/Default_diagonalize_traits.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/boost/graph/alpha_expansion_graphcut.h>
#include <CGAL/point_generators_3.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for_each.h>
#endif

#include <unordered_map>
#include <vector>

namespace CGAL {

template<typename GeomTraits, typename PointRange, typename PointMap, typename NormalMap>
class Wasure {
public:
  using Geom_traits = GeomTraits;
  using Point_range = PointRange;
  using Point_map = PointMap;
  using Normal_map = NormalMap;
  using FT = typename Geom_traits::FT;
  using Point_3 = typename Geom_traits::Point_3;
  using Vector_3 = typename Geom_traits::Vector_3;

private:
  using Search_traits_base = Search_traits_3<Geom_traits>;
  using Search_traits = Search_traits_adapter<std::uint32_t, Point_map, Search_traits_base>;
  using Splitter = Sliding_midpoint<Search_traits>;
  using Distance = Distance_adapter<std::uint32_t, Point_map, Euclidean_distance<Search_traits_base>>;
  using Tree = Kd_tree<Search_traits, Splitter, Tag_true, Tag_true>;
  using Knn = Orthogonal_k_neighbor_search<Search_traits, Distance, Splitter, Tree>;

  using DT = CGAL::Delaunay_triangulation_3<Geom_traits>;
  using Cell_handle = typename DT::Cell_handle;
  using Vertex_handle = typename DT::Vertex_handle;
  using Facet = typename DT::Facet;
  using Locate_type = typename DT::Locate_type;

  using Point_generator = CGAL::Random_points_in_tetrahedron_3<Point_3>;

  Point_range& m_points;
  Point_map m_point_map;
  Normal_map m_normal_map;
  Bbox_3 m_bbox;
  std::vector<std::uint32_t> m_simplified;
  std::vector<std::array<Vector_3, 3>> m_eigen_vectors;
  std::vector<std::array<double, 3>> m_eigen_values;
  std::vector<Vector_3> m_origin;

  Tree m_tree;
  DT m_triangulation;
  std::vector<Cell_handle> m_tets;
  std::unordered_map<Cell_handle, std::size_t> m_tet_indices;
  std::vector<std::pair<std::size_t, std::size_t>> m_edges;
  std::vector<double> m_edge_weights;
  std::vector<std::array<double, 3>> m_mass_function; // empty, occupied, unknown
  std::vector<double> m_tet_volumes;

  bool m_has_normals;
  bool m_has_point_origin;
public:
  Wasure(PointRange &points, Point_map point_map, Normal_map normal_map)
    : m_points(points), m_point_map(point_map), m_normal_map(normal_map),
      m_tree(boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(points.size()), Splitter(), Search_traits(point_map)),
      m_bbox(CGAL::bbox_3(CGAL::make_transform_iterator_from_property_map(m_points.begin(), m_point_map), CGAL::make_transform_iterator_from_property_map(m_points.end(), m_point_map))),
      m_has_normals(true), m_has_point_origin(false)
  {}
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
      std::array<FT, 3> sum = {0, 0, 0};
      Knn search(m_tree, get(m_point_map, idx), 150);
      for (typename Knn::iterator it = search.begin(); it != search.end(); ++it) {
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

        pca<DiagonalizeTraits>(pts, Point_3(sum[0]/pts.size(), sum[1]/pts.size(), sum[2]/pts.size()), eigen_values, eigen_vectors);

        double cur_entropy = 0;

        std::array<FT, 3> ev = {.0, .0, .0};

        // Normalization like in reference implementation
        double sum = eigen_values[0] + eigen_values[1] + eigen_values[2];
        ev[0] = (eigen_values[0] - eigen_values[1]) / sum;
        ev[1] = (eigen_values[1] - eigen_values[2]) / sum;
        ev[2] = eigen_values[2] / sum;

        for (int i = 0;i<3;i++)
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

    for (auto idx : m_points) {
      Point_3 p = get(m_point_map, idx);
      Vector_3 n = get(m_normal_map, idx);
      Vector_3 np = Vector_3(p.x(), p.y(), p.z());
      assert(n * np > 0);
      Vector_3 ev = m_eigen_vectors[idx][2];
      assert(n * ev > 0);
      assert(np * ev > 0);
    }
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
      Knn search(m_tree, p, 50);
      for (typename Knn::iterator it = search.begin() + 1; it != search.end(); ++it) {
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

          if (!m_origin.empty()) {
            los[idx][0] += m_origin[idx].x();
            los[idx][1] += m_origin[idx].y();
            los[idx][2] += m_origin[idx].z();
          }
        }
      }
      std::cout << count[5] << std::endl;

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
          if (!m_origin.empty() && random.uniform_01<double>() > 0.8) {
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
      Point_3 p;
      if (i == 0)
        p = CGAL::centroid(ch->vertex(0)->point(), ch->vertex(1)->point(), ch->vertex(2)->point(), ch->vertex(3)->point());
      else
        p = *pg++;

      // Compute local mass for each sample
      Knn search(m_tree, p, 30);
      std::array<double, 3> sample_mass = { 0, 0, 1 };
      for (typename Knn::iterator it = search.begin(); it != search.end(); ++it) {
        Point_3 s = get(m_point_map, it->first);
        double largest_ev = m_eigen_values[it->first][0];
        std::array<double, 3> point_mass = { 0, 0, 0 };
        std::array<double, 3> point_coefficients = calculate_point_coefficients(s, p, m_eigen_vectors[it->first]);
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

  void orient_features() {
    if (m_has_normals) {
      for (auto idx : m_points) {
        Vector_3 normal = get(m_normal_map, idx);
        if (normal * m_eigen_vectors[idx][2] < 0)
          m_eigen_vectors[idx][2] = -m_eigen_vectors[idx][2];
      }
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
};

} // namespace CGAL

#endif
