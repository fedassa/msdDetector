
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(MSD_DETECTOR)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type default to Release" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

# postfix, based on type
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "postfix applied to debug build of libraries")
SET(CMAKE_RELEASE_POSTFIX "" CACHE STRING "postfix applied to release build of libraries")
SET(CMAKE_RELWITHDEBINFO_POSTFIX "rd" CACHE STRING "postfix applied to release-with-debug-information libraries")
SET(CMAKE_MINSIZEREL_POSTFIX "s" CACHE STRING "postfix applied to minimium-size-build libraries")

###########################################################################################
MSD_DETECTOR_SET_OUTPUT_DIR()

# Allow the developer to select if Dynamic or Static libraries are built
OPTION(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
IF(BUILD_SHARED_LIBS)
  SET(MSD_DETECTOR_LIB_TYPE SHARED)
ELSE(BUILD_SHARED_LIBS)
  SET(MSD_DETECTOR_LIB_TYPE STATIC)
ENDIF(BUILD_SHARED_LIBS)

###########################################################################################
# Special flags
set(MSD_DETECTOR_C_FLAGS)
set(MSD_DETECTOR_CXX_FLAGS)
set(MSD_DETECTOR_CXX_FLAGS_RELEASE)
set(MSD_DETECTOR_EXE_LINKER_FLAGS)
set(MSD_DETECTOR_EXE_LINKER_FLAGS_RELEASE)
set(MSD_DETECTOR_SHARED_LINKER_FLAGS)
set(MSD_DETECTOR_SHARED_LINKER_FLAGS_RELEASE)
if(CMAKE_COMPILER_IS_GNUCC)
  set(MSD_DETECTOR_CXX_FLAGS "${QONSULT_CXX_FLAGS} -fexceptions -Wall -Wextra")
  set(MSD_DETECTOR_C_FLAGS "${QONSULT_C_FLAGS} -Wall -Wextra")
  set(MSD_DETECTOR_EXE_LINKER_FLAGS "${QONSULT_EXE_LINKER_FLAGS} -Wl,--warn-unresolved-symbols -Wl,--as-needed")
  set(MSD_DETECTOR_SHARED_LINKER_FLAGS "${QONSULT_SHARED_LINKER_FLAGS} -Wl,--warn-unresolved-symbols -Wl,--as-needed")
elseif(MSVC)
  add_definitions ("-DBOOST_ALL_NO_LIB -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX")
  #SET(QONSULT_CXX_FLAGS "${QONSULT_CXX_FLAGS} /bigobj /EHsc /fp:fast /wd4800")
  SET(MSD_DETECTOR_CXX_FLAGS "${QONSULT_CXX_FLAGS} /EHsc /fp:fast /wd4800 /W4")
  SET(MSD_DETECTOR_CXX_FLAGS_RELEASE "${QONSULT_CXX_FLAGS_RELEASE} /GL /wd4800 /W4")
  SET(MSD_DETECTOR_SHARED_LINKER_FLAGS_RELEASE "${QONSULT_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
  SET(MSD_DETECTOR_EXE_LINKER_FLAGS_RELEASE "${QONSULT_EXE_LINKER_FLAGS_RELEASE} /LTCG")
  if(MSVC90 OR MSVC10)
    SET(MSD_DETECTOR_CXX_FLAGS "${MSD_DETECTOR_CXX_FLAGS} /MP")
  endif(MSVC90 OR MSVC10)
endif(CMAKE_COMPILER_IS_GNUCC)

# SSE Check
MSD_DETECTOR_CHECK_FOR_SSE()

if(MSVC90)
  add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY=1)
  message("Disabling memory alignment for MSVC8")
endif(MSVC90)

option(USE_PROJECT_FOLDERS "Use folders to organize MSD_DETECTOR projects in an IDE." OFF)
#mark_as_advanced(USE_PROJECT_FOLDERS)
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif(USE_PROJECT_FOLDERS)

###########################################################################################
# Find modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Boost
set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost 1.49 COMPONENTS thread system)
if(Boost_FOUND)
 add_definitions(-DBOOST_ALL_DYN_LINK)
 add_definitions(-DBOOST_MULTICORE)
 include_directories(${Boost_INCLUDE_DIRS})
 link_directories(${Boost_LIBRARY_DIRS})
else(Boost_FOUND)
  set(BOOST_LIBRARYDIR "" CACHE PATH "Root of Boost installation")
endif(Boost_FOUND)

# #OpenCV
find_package(OpenCV 3.0 REQUIRED highgui core imgcodecs imgproc)
	if(OpenCV_INCLUDE_DIRS)
		include_directories(${OpenCV_INCLUDE_DIRS})
		link_directories(${OpenCV_LIB_DIRS})
	endif(OpenCV_INCLUDE_DIRS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# TARGETS
ADD_SUBDIRECTORY(cmake/msdDetector)


	
	

