#pragma once
#include "csr.h"
#include "op.h"

void mat_dot(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output);
void mat_transpose(array2d_t<float>& input1, array2d_t<float>& output);
void mat_add(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output);