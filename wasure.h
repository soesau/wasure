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

//#include <CGAL/Kd_tree.h>
#include <CGAL/Splitters.h>
#include <CGAL/Default_diagonalize_traits.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Point_set_3.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for_each.h>
#endif

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

  Point_range& m_points;
  Point_map m_point_map;
  Normal_map m_normal_map;
  std::vector<std::uint32_t> m_simplified;
  std::vector<std::array<Vector_3, 3>> m_eigen_vectors;
  std::vector<std::array<FT, 3>> m_eigen_values;
  Tree m_tree;

public:
  Wasure(PointRange &points, Point_map point_map, Normal_map normal_map)
    : m_points(points), m_point_map(point_map), m_normal_map(normal_map),
      m_tree(boost::counting_iterator<std::uint32_t>(0), boost::counting_iterator<std::uint32_t>(points.size()), Splitter(), Search_traits(point_map)) {}
  ~Wasure() {}

  template<typename Concurrency_tag = Sequential_tag, typename DiagonalizeTraits = CGAL::Default_diagonalize_traits<double, 3>>
  void compute_features() {
    std::vector<Point_3> pts;
    pts.reserve(150);

    double entropy = 1000000000;

    for (auto idx : m_points) {
      pts.clear();
      std::array<FT, 3> sum = {0, 0, 0};
      Knn search(m_tree, get(m_point_map, idx), 150);
      for (typename Knn::iterator it = search.begin(); it != search.end(); ++it) {
        pts.push_back(get(m_point_map, it->first));

        sum[0] += pts.back().x();
        sum[1] += pts.back().y();
        sum[2] += pts.back().z();

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
    }
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
  void pca(const std::vector<Point_3> &pts, const Point_3 &mean, std::array<FT, 3> &eigen_values, std::array<Vector_3, 3> &eigen_vectors) {
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

    for (int i = 0;i<3;i++)
      if (eigen_values[i] < 0.0000001 || eigen_values[i] != eigen_values[i])
        eigen_values[i] = 0.0000001;

    for (int i = 0;i<3;i++)
      eigen_vectors[i] = Vector_3(evectors[3 * i], evectors[3 * i + 1], evectors[3 * i + 2]);
  }
};

} // namespace CGAL

#endif
