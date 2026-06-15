cmake_minimum_required(VERSION 3.31)
project(CGALWasure)

find_package(CGAL REQUIRED)
find_package(TBB REQUIRED)
include(CGAL_TBB_support)

find_package(LASLIB QUIET)
include(CGAL_LASLIB_support)

create_single_source_cgal_program("wasure.cpp")

if(TARGET CGAL::TBB_support)
  target_link_libraries(wasure PRIVATE CGAL::TBB_support)
endif()

if(TARGET CGAL::LASLIB_support)
  target_link_libraries(wasure PRIVATE CGAL::LASLIB_support)
endif()
