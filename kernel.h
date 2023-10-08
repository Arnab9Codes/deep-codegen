#pragma once
#include "csr.h"
#include "op.h"


void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm);
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse);
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse);
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse);
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse);
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse);
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse);
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse);
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse);
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse);
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse);
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t);
void mat_dot(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int P, int N);
void mat_transpose(array2d_t<float>& input1, array2d_t<float>& output1, int M, int N);
void mat_add(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int N);
void mat_mul(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, int M, int N);
void mat_norm(array2d_t<float>& input1, int M, int N);
