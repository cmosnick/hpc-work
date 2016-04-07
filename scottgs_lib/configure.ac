AC_PREREQ(2.59)
AC_INIT([libscottgs], [0.0.0], [GrantScott@missouri.edu])
AC_CONFIG_AUX_DIR([config])

AM_INIT_AUTOMAKE(libscottgs,0.0.0)

AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LN_S
AC_PROG_MAKE_SET

AC_LANG([C++])

AC_CHECK_HEADER([boost/filesystem.hpp], [], AC_MSG_ERROR([boost/filesystem.hpp not found]))
AC_CHECK_HEADER([boost/iterator/iterator_facade.hpp], [], AC_MSG_ERROR([boost/iterator/iterator_facade.hpp not found]))
AC_CHECK_HEADER([boost/mpl/if.hpp], [], AC_MSG_ERROR([boost/mpl/if.hpp not found]))
AC_CHECK_HEADER([boost/numeric/ublas/matrix.hpp], [], AC_MSG_ERROR([boost/numeric/ublas/matrix.hpp not found]))
AC_CHECK_HEADER([boost/numeric/ublas/matrix_proxy.hpp], [], AC_MSG_ERROR([boost/numeric/ublas/matrix_proxy.hpp not found]))
AC_CHECK_HEADER([boost/regex.hpp], [], AC_MSG_ERROR([boost/regex.hpp not found]))
AC_CHECK_HEADER([boost/type_traits/is_const.hpp], [], AC_MSG_ERROR([boost/type_traits/is_const.hpp not found]))
AC_CHECK_HEADER([boost/utility.hpp], [], AC_MSG_ERROR([boost/utility.hpp not found]))

AC_CHECK_LIB([boost_filesystem-mt], [main], [], AC_MSG_ERROR([libboost_filesystem-mt not found]))
AC_CHECK_LIB([boost_regex-mt], [main], [], AC_MSG_ERROR([libboost_regex-mt not found]))

AM_CXXFLAGS="-Wall -Wextra -g -rdynamic"
AC_SUBST([AM_CXXFLAGS])

AC_CONFIG_FILES([
                 Makefile
                 doc/Makefile
                 src/Makefile
                 ])

AC_OUTPUT
