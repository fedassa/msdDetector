
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type default to Release" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

###########################################################################################
# TARGETS

SET(SRCS ${PROJECT_SOURCE_DIR}/include/msdImgPyramid.h
	${PROJECT_SOURCE_DIR}/include/msd.h
	${PROJECT_SOURCE_DIR}/src/msdImgPyramid.cpp
	${PROJECT_SOURCE_DIR}/src/msd.cpp
	${PROJECT_SOURCE_DIR}/src/msdTest.cpp
	)
	
MSD_DETECTOR_ADD_EXECUTABLE(msdDetector FILES ${SRCS} LINK_WITH ${Boost_LIBRARIES} ${OpenCV_LIBS})

