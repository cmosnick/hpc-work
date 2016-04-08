#include <exception>
#include <stdexcept>
#include <map>
#include <list>
#include <string>
#include <sstream>
#include <iostream>

// +++++++++++++++++++++++++++++++++++++++++++++++++++

#include <boost/filesystem.hpp>

// See the boost documentation for the filesystem
// Especially: http://www.boost.org/doc/libs/1_41_0/libs/filesystem/doc/reference.html#Path-decomposition-table
// Link against boost_filesystem-mt (for multithreaded) or boost_filesystem
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

// Define my map keys
#define REG_FILE   "REGULAR"
#define DIR_FILE   "DIRECTORY"
#define OTHER_FILE "OTHER"

// +++++++++++++++++++++++++++++++++++++++++++++++++++

namespace scottgs {

    // ===== DEFINITIONS =======
    typedef std::list<boost::filesystem::path> path_list_type;
    std::map<std::string,path_list_type> getFiles(boost::filesystem::path dir);

};