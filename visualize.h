#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template<class T>
void graph_out(const std::vector<T>& fragments, const std::string & filename = "graph.dot") {
  std::string graph_type = "graph";
  std::string graph_name = "loop_tree";
  std::string root_color = "red";
  std::string node_color = "blue";
  std::string node_shape = "circle";
  std::string label = "\"\"";

  std::ofstream fout(filename.c_str());

  std::vector<int> root_node;
  for (int i = 0; i < fragments.size(); ++i) {
    if (fragments[i].is_root()) root_node.push_back(i);
  }

  fout << graph_type << " " << graph_name << " {" << std::endl;
  // properties of root nodes
  fout << "node [shape=" << node_shape << ", style=filled, fillcolor=" << root_color << "]";
  for (int i = 0; i < root_node.size(); ++i) {
    fout << " " << root_node[i];
  }
  fout << ";" << std::endl;
  // properties of the other nodes
  fout << "node [shape=" << node_shape << ", style=filled, fillcolor=" << node_color << "]";
  fout << ";" << std::endl;
    
  for (int i = 0; i < fragments.size(); ++i) {
    if(!fragments[i].is_root()) {
      fout << i << " -- " << fragments[i].parent() << ";" << std::endl;
    }
  }
  fout << "}" << std::endl;
  fout.close();
}
