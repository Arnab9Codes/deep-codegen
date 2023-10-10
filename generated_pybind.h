inline void export_kernel(py::module &m) { 
    m.def("mat_dot",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return mat_dot(input1_array, input2_array, output_array);
    }
  );
    m.def("mat_transpose",[](py::capsule& input1, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return mat_transpose(input1_array, output_array);
    }
  );
    m.def("mat_add",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return mat_add(input1_array, input2_array, output_array);
    }
  );
}