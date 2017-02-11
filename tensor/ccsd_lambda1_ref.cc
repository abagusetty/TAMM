//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include <iostream>
#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"

/*
 *  lambda1 {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12 = O;
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9 = V;
 *
 *  array i0[O][V];
 *  array f[N][N]: irrep_f;
 *  array y_ov[O][V]: irrep_y;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *  array lambda1_15_2_1[O,O][O,V];
 *  array lambda1_3_3_1[O][V];
 *  array lambda1_5_5_1[O][V];
 *  array lambda1_5_3_1[V][V];
 *  array lambda1_5_2_1[O][O];
 *  array lambda1_6_4_1[O][V];
 *  array lambda1_6_2_2_1[O,O][O,V];
 *  array lambda1_3_1[V][V];
 *  array lambda1_2_1[O][O];
 *  array lambda1_6_2_1[O,O][O,O];
 *  array lambda1_10_1[O][O];
 *  array lambda1_6_3_1[O,V][O,V];
 *  array lambda1_8_1[V][O];
 *  array lambda1_7_1[V,V][O,V];
 *  array lambda1_5_2_2_1[O][V];
 *  array lambda1_6_1[O,V][O,O];
 *  array lambda1_6_5_1[O,O][O,V];
 *  array lambda1_13_2_2_1[O,O][O,V];
 *  array lambda1_11_1[V][V];
 *  array lambda1_5_1[V][O];
 *  array lambda1_8_4_1[O,O][O,V];
 *  array lambda1_13_2_1[O,O][O,O];
 *  array lambda1_14_1[O,O][O,O];
 *  array lambda1_13_3_1[O,O][O,V];
 *  array lambda1_15_1[O,V][O,V];
 *  array lambda1_8_3_1[O][O];
 *  array lambda1_14_2_1[O,O][O,V];
 *  array lambda1_12_1[O,O][O,V];
 *  array lambda1_13_4_1[O][O];
 *  array lambda1_5_6_1[O,O][O,V];
 *  array lambda1_9_1[O][O];
 *  array lambda1_2_2_1[O][V];
 *  array lambda1_13_1[O,V][O,O];
 *
 *  lambda1_1:       i0[h2,p1] += 1 * f[h2,p1];
 *  lambda1_2_1:     lambda1_2_1[h2,h7] += 1 * f[h2,h7];
 *  lambda1_2_2_1:   lambda1_2_2_1[h2,p3] += 1 * f[h2,p3];
 *  lambda1_2_2_2:   lambda1_2_2_1[h2,p3] += 1 * t_vo[p5,h6] * v[h2,h6,p3,p5];
 *  lambda1_2_2:     lambda1_2_1[h2,h7] += 1 * t_vo[p3,h7] *
 * lambda1_2_2_1[h2,p3];
 *  lambda1_2_3:     lambda1_2_1[h2,h7] += 1 * t_vo[p3,h4] * v[h2,h4,h7,p3];
 *  lambda1_2_4:     lambda1_2_1[h2,h7] += -1/2 * t_vvoo[p3,p4,h6,h7] *
 * v[h2,h6,p3,p4];
 *  lambda1_2:       i0[h2,p1] += -1 * y_ov[h7,p1] * lambda1_2_1[h2,h7];
 *  lambda1_3_1:     lambda1_3_1[p7,p1] += 1 * f[p7,p1];
 *  lambda1_3_2:     lambda1_3_1[p7,p1] += -1 * t_vo[p3,h4] * v[h4,p7,p1,p3];
 *  lambda1_3_3_1:   lambda1_3_3_1[h4,p1] += 1 * t_vo[p5,h6] * v[h4,h6,p1,p5];
 *  lambda1_3_3:     lambda1_3_1[p7,p1] += -1 * t_vo[p7,h4] *
 * lambda1_3_3_1[h4,p1];
 *  lambda1_3:       i0[h2,p1] += 1 * y_ov[h2,p7] * lambda1_3_1[p7,p1];
 *  lambda1_4:       i0[h2,p1] += -1 * y_ov[h4,p3] * v[h2,p3,h4,p1];
 *  lambda1_5_1:     lambda1_5_1[p9,h11] += 1 * f[p9,h11];
 *  lambda1_5_2_1:   lambda1_5_2_1[h10,h11] += 1 * f[h10,h11];
 *  lambda1_5_2_2_1: lambda1_5_2_2_1[h10,p3] += 1 * f[h10,p3];
 *  lambda1_5_2_2_2: lambda1_5_2_2_1[h10,p3] += -1 * t_vo[p7,h8] *
 * v[h8,h10,p3,p7];
 *  lambda1_5_2_2:   lambda1_5_2_1[h10,h11] += 1 * t_vo[p3,h11] *
 * lambda1_5_2_2_1[h10,p3];
 *  lambda1_5_2_3:   lambda1_5_2_1[h10,h11] += -1 * t_vo[p5,h6] *
 * v[h6,h10,h11,p5];
 *  lambda1_5_2_4:   lambda1_5_2_1[h10,h11] += 1/2 * t_vvoo[p3,p4,h6,h11] *
 * v[h6,h10,p3,p4];
 *  lambda1_5_2:     lambda1_5_1[p9,h11] += -1 * t_vo[p9,h10] *
 * lambda1_5_2_1[h10,h11];
 *  lambda1_5_3_1:   lambda1_5_3_1[p9,p7] += 1 * f[p9,p7];
 *  lambda1_5_3_2:   lambda1_5_3_1[p9,p7] += 1 * t_vo[p5,h6] * v[h6,p9,p5,p7];
 *  lambda1_5_3:     lambda1_5_1[p9,h11] += 1 * t_vo[p7,h11] *
 * lambda1_5_3_1[p9,p7];
 *  lambda1_5_4:     lambda1_5_1[p9,h11] += -1 * t_vo[p3,h4] * v[h4,p9,h11,p3];
 *  lambda1_5_5_1:   lambda1_5_5_1[h5,p4] += 1 * f[h5,p4];
 *  lambda1_5_5_2:   lambda1_5_5_1[h5,p4] += 1 * t_vo[p7,h8] * v[h5,h8,p4,p7];
 *  lambda1_5_5:     lambda1_5_1[p9,h11] += 1 * t_vvoo[p4,p9,h5,h11] *
 * lambda1_5_5_1[h5,p4];
 *  lambda1_5_6_1:   lambda1_5_6_1[h5,h6,h11,p4] += 1 * v[h5,h6,h11,p4];
 *  lambda1_5_6_2:   lambda1_5_6_1[h5,h6,h11,p4] += -1 * t_vo[p7,h11] *
 * v[h5,h6,p4,p7];
 *  lambda1_5_6:     lambda1_5_1[p9,h11] += 1/2 * t_vvoo[p4,p9,h5,h6] *
 * lambda1_5_6_1[h5,h6,h11,p4];
 *  lambda1_5_7:     lambda1_5_1[p9,h11] += 1/2 * t_vvoo[p3,p4,h6,h11] *
 * v[h6,p9,p3,p4];
 *  lambda1_5:       i0[h2,p1] += 1 * y_oovv[h2,h11,p1,p9] *
 * lambda1_5_1[p9,h11];
 *  lambda1_6_1:     lambda1_6_1[h2,p9,h11,h12] += 1 * v[h2,p9,h11,h12];
 *  lambda1_6_2_1:   lambda1_6_2_1[h2,h7,h11,h12] += 1 * v[h2,h7,h11,h12];
 *  lambda1_6_2_2_1: lambda1_6_2_2_1[h2,h7,h12,p3] += 1 * v[h2,h7,h12,p3];
 *  lambda1_6_2_2_2: lambda1_6_2_2_1[h2,h7,h12,p3] += -1/2 * t_vo[p5,h12] *
 * v[h2,h7,p3,p5];
 *  lambda1_6_2_2:   lambda1_6_2_1[h2,h7,h11,h12] += -2 * t_vo[p3,h11] *
 * lambda1_6_2_2_1[h2,h7,h12,p3];
 *  lambda1_6_2_3:   lambda1_6_2_1[h2,h7,h11,h12] += 1/2 * t_vvoo[p3,p4,h11,h12]
 * * v[h2,h7,p3,p4];
 *  lambda1_6_2:     lambda1_6_1[h2,p9,h11,h12] += -1 * t_vo[p9,h7] *
 * lambda1_6_2_1[h2,h7,h11,h12];
 *  lambda1_6_3_1:   lambda1_6_3_1[h2,p9,h12,p3] += 1 * v[h2,p9,h12,p3];
 *  lambda1_6_3_2:   lambda1_6_3_1[h2,p9,h12,p3] += -1/2 * t_vo[p5,h12] *
 * v[h2,p9,p3,p5];
 *  lambda1_6_3:     lambda1_6_1[h2,p9,h11,h12] += -2 * t_vo[p3,h11] *
 * lambda1_6_3_1[h2,p9,h12,p3];
 *  lambda1_6_4_1:   lambda1_6_4_1[h2,p5] += 1 * f[h2,p5];
 *  lambda1_6_4_2:   lambda1_6_4_1[h2,p5] += 1 * t_vo[p7,h8] * v[h2,h8,p5,p7];
 *  lambda1_6_4:     lambda1_6_1[h2,p9,h11,h12] += 1 * t_vvoo[p5,p9,h11,h12] *
 * lambda1_6_4_1[h2,p5];
 *  lambda1_6_5_1:   lambda1_6_5_1[h2,h6,h12,p4] += 1 * v[h2,h6,h12,p4];
 *  lambda1_6_5_2:   lambda1_6_5_1[h2,h6,h12,p4] += -1 * t_vo[p7,h12] *
 * v[h2,h6,p4,p7];
 *  lambda1_6_5:     lambda1_6_1[h2,p9,h11,h12] += -2 * t_vvoo[p4,p9,h6,h11] *
 * lambda1_6_5_1[h2,h6,h12,p4];
 *  lambda1_6_6:     lambda1_6_1[h2,p9,h11,h12] += 1/2 * t_vvoo[p3,p4,h11,h12] *
 * v[h2,p9,p3,p4];
 *  lambda1_6:       i0[h2,p1] += -1/2 * y_oovv[h11,h12,p1,p9] *
 * lambda1_6_1[h2,p9,h11,h12];
 *  lambda1_7_1:     lambda1_7_1[p5,p8,h7,p1] += -1 * v[p5,p8,h7,p1];
 *  lambda1_7_2:     lambda1_7_1[p5,p8,h7,p1] += 1 * t_vo[p3,h7] *
 * v[p5,p8,p1,p3];
 *  lambda1_7:       i0[h2,p1] += 1/2 * y_oovv[h2,h7,p5,p8] *
 * lambda1_7_1[p5,p8,h7,p1];
 *  lambda1_8_1:     lambda1_8_1[p9,h10] += 1 * t_vo[p9,h10];
 *  lambda1_8_2:     lambda1_8_1[p9,h10] += 1 * t_vvoo[p3,p9,h5,h10] *
 * y_ov[h5,p3];
 *  lambda1_8_3_1:   lambda1_8_3_1[h6,h10] += 1 * t_vo[p5,h10] * y_ov[h6,p5];
 *  lambda1_8_3_2:   lambda1_8_3_1[h6,h10] += 1/2 * t_vvoo[p3,p4,h5,h10] *
 * y_oovv[h5,h6,p3,p4];
 *  lambda1_8_3:     lambda1_8_1[p9,h10] += -1 * t_vo[p9,h6] *
 * lambda1_8_3_1[h6,h10];
 *  lambda1_8_4_1:   lambda1_8_4_1[h5,h6,h10,p3] += 1 * t_vo[p7,h10] *
 * y_oovv[h5,h6,p3,p7];
 *  lambda1_8_4:     lambda1_8_1[p9,h10] += -1/2 * t_vvoo[p3,p9,h5,h6] *
 * lambda1_8_4_1[h5,h6,h10,p3];
 *  lambda1_8:       i0[h2,p1] += 1 * lambda1_8_1[p9,h10] * v[h2,h10,p1,p9];
 *  lambda1_9_1:     lambda1_9_1[h2,h3] += 1 * t_vo[p4,h3] * y_ov[h2,p4];
 *  lambda1_9_2:     lambda1_9_1[h2,h3] += 1/2 * t_vvoo[p4,p5,h3,h6] *
 * y_oovv[h2,h6,p4,p5];
 *  lambda1_9:       i0[h2,p1] += -1 * lambda1_9_1[h2,h3] * f[h3,p1];
 *  lambda1_10_1:    lambda1_10_1[h6,h8] += 1 * t_vo[p3,h8] * y_ov[h6,p3];
 *  lambda1_10_2:    lambda1_10_1[h6,h8] += 1/2 * t_vvoo[p3,p4,h5,h8] *
 * y_oovv[h5,h6,p3,p4];
 *  lambda1_10:      i0[h2,p1] += 1 * lambda1_10_1[h6,h8] * v[h2,h8,h6,p1];
 *  lambda1_11_1:    lambda1_11_1[p7,p8] += 1 * t_vo[p7,h4] * y_ov[h4,p8];
 *  lambda1_11_2:    lambda1_11_1[p7,p8] += 1/2 * t_vvoo[p3,p7,h5,h6] *
 * y_oovv[h5,h6,p3,p8];
 *  lambda1_11:      i0[h2,p1] += 1 * lambda1_11_1[p7,p8] * v[h2,p8,p1,p7];
 *  lambda1_12_1:    lambda1_12_1[h2,h6,h4,p5] += 1 * t_vo[p3,h4] *
 * y_oovv[h2,h6,p3,p5];
 *  lambda1_12:      i0[h2,p1] += 1 * lambda1_12_1[h2,h6,h4,p5] *
 * v[h4,p5,h6,p1];
 *  lambda1_13_1:    lambda1_13_1[h2,p9,h6,h12] += -1 * t_vvoo[p3,p9,h6,h12] *
 * y_ov[h2,p3];
 *  lambda1_13_2_1:  lambda1_13_2_1[h2,h10,h6,h12] += -1 * t_vvoo[p3,p4,h6,h12]
 * * y_oovv[h2,h10,p3,p4];
 *  lambda1_13_2_2_1:lambda1_13_2_2_1[h2,h10,h6,p5] += 1 * t_vo[p7,h6] *
 * y_oovv[h2,h10,p5,p7];
 *  lambda1_13_2_2:  lambda1_13_2_1[h2,h10,h6,h12] += 2 * t_vo[p5,h12] *
 * lambda1_13_2_2_1[h2,h10,h6,p5];
 *  lambda1_13_2:    lambda1_13_1[h2,p9,h6,h12] += -1/2 * t_vo[p9,h10] *
 * lambda1_13_2_1[h2,h10,h6,h12];
 *  lambda1_13_3_1:  lambda1_13_3_1[h2,h5,h6,p3] += 1 * t_vo[p7,h6] *
 * y_oovv[h2,h5,p3,p7];
 *  lambda1_13_3:    lambda1_13_1[h2,p9,h6,h12] += 2 * t_vvoo[p3,p9,h5,h12] *
 * lambda1_13_3_1[h2,h5,h6,p3];
 *  lambda1_13_4_1:  lambda1_13_4_1[h2,h12] += 1 * t_vvoo[p3,p4,h5,h12] *
 * y_oovv[h2,h5,p3,p4];
 *  lambda1_13_4:    lambda1_13_1[h2,p9,h6,h12] += -1 * t_vo[p9,h6] *
 * lambda1_13_4_1[h2,h12];
 *  lambda1_13:      i0[h2,p1] += 1/2 * lambda1_13_1[h2,p9,h6,h12] *
 * v[h6,h12,p1,p9];
 *  lambda1_14_1:    lambda1_14_1[h2,h7,h6,h8] += -1 * t_vvoo[p3,p4,h6,h8] *
 * y_oovv[h2,h7,p3,p4];
 *  lambda1_14_2_1:  lambda1_14_2_1[h2,h7,h6,p3] += 1 * t_vo[p5,h6] *
 * y_oovv[h2,h7,p3,p5];
 *  lambda1_14_2:    lambda1_14_1[h2,h7,h6,h8] += 2 * t_vo[p3,h8] *
 * lambda1_14_2_1[h2,h7,h6,p3];
 *  lambda1_14:      i0[h2,p1] += 1/4 * lambda1_14_1[h2,h7,h6,h8] *
 * v[h6,h8,h7,p1];
 *  lambda1_15_1:    lambda1_15_1[h2,p8,h6,p7] += 1 * t_vvoo[p3,p8,h5,h6] *
 * y_oovv[h2,h5,p3,p7];
 *  lambda1_15_2_1:  lambda1_15_2_1[h2,h4,h6,p7] += 1 * t_vo[p5,h6] *
 * y_oovv[h2,h4,p5,p7];
 *  lambda1_15_2:    lambda1_15_1[h2,p8,h6,p7] += -1 * t_vo[p8,h4] *
 * lambda1_15_2_1[h2,h4,h6,p7];
 *  lambda1_15:      i0[h2,p1] += 1 * lambda1_15_1[h2,p8,h6,p7] *
 * v[h6,p7,p1,p8];
 *
 *  }
*/

extern "C" {
void ccsd_lambda1_1_(Integer *d_f, Integer *k_f_offset, Integer *d_i0,
                     Integer *k_i0_offset);
void ccsd_lambda1_2_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda1_2_1, Integer *k_lambda1_2_1_offset);
void ccsd_lambda1_2_2_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda1_2_2_1,
                         Integer *k_lambda1_2_2_1_offset);
void ccsd_lambda1_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_2_2_1,
                         Integer *k_lambda1_2_2_1_offset);
void ccsd_lambda1_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_2_2_1,
                       Integer *k_lambda1_2_2_1_offset, Integer *d_lambda1_2_1,
                       Integer *k_lambda1_2_1_offset);
void ccsd_lambda1_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda1_2_1,
                       Integer *k_lambda1_2_1_offset);
void ccsd_lambda1_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda1_2_1, Integer *k_lambda1_2_1_offset);
void ccsd_lambda1_2_(Integer *d_y_ov, Integer *k_y_ov_offset,
                     Integer *d_lambda1_2_1, Integer *k_lambda1_2_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_3_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda1_3_1, Integer *k_lambda1_3_1_offset);
void ccsd_lambda1_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda1_3_1,
                       Integer *k_lambda1_3_1_offset);
void ccsd_lambda1_3_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_3_3_1,
                         Integer *k_lambda1_3_3_1_offset);
void ccsd_lambda1_3_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_3_3_1,
                       Integer *k_lambda1_3_3_1_offset, Integer *d_lambda1_3_1,
                       Integer *k_lambda1_3_1_offset);
void ccsd_lambda1_3_(Integer *d_y_ov, Integer *k_y_ov_offset,
                     Integer *d_lambda1_3_1, Integer *k_lambda1_3_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_4_(Integer *d_y_ov, Integer *k_y_ov_offset, Integer *d_v,
                     Integer *k_v_offset, Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_5_1_(Integer *d_f, Integer *k_f_offset,
                       Integer *d_lambda1_5_1, Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_2_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda1_5_2_1,
                         Integer *k_lambda1_5_2_1_offset);
void ccsd_lambda1_5_2_2_1_(Integer *d_f, Integer *k_f_offset,
                           Integer *d_lambda1_5_2_2_1,
                           Integer *k_lambda1_5_2_2_1_offset);
void ccsd_lambda1_5_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_v, Integer *k_v_offset,
                           Integer *d_lambda1_5_2_2_1,
                           Integer *k_lambda1_5_2_2_1_offset);
void ccsd_lambda1_5_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                         Integer *d_lambda1_5_2_2_1,
                         Integer *k_lambda1_5_2_2_1_offset,
                         Integer *d_lambda1_5_2_1,
                         Integer *k_lambda1_5_2_1_offset);
void ccsd_lambda1_5_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_5_2_1,
                         Integer *k_lambda1_5_2_1_offset);
void ccsd_lambda1_5_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                         Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_5_2_1,
                         Integer *k_lambda1_5_2_1_offset);
void ccsd_lambda1_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_5_2_1,
                       Integer *k_lambda1_5_2_1_offset, Integer *d_lambda1_5_1,
                       Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_3_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda1_5_3_1,
                         Integer *k_lambda1_5_3_1_offset);
void ccsd_lambda1_5_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_5_3_1,
                         Integer *k_lambda1_5_3_1_offset);
void ccsd_lambda1_5_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_5_3_1,
                       Integer *k_lambda1_5_3_1_offset, Integer *d_lambda1_5_1,
                       Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_4_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda1_5_1,
                       Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_5_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda1_5_5_1,
                         Integer *k_lambda1_5_5_1_offset);
void ccsd_lambda1_5_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_5_5_1,
                         Integer *k_lambda1_5_5_1_offset);
void ccsd_lambda1_5_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_lambda1_5_5_1,
                       Integer *k_lambda1_5_5_1_offset, Integer *d_lambda1_5_1,
                       Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_6_1_(Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_5_6_1,
                         Integer *k_lambda1_5_6_1_offset);
void ccsd_lambda1_5_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_5_6_1,
                         Integer *k_lambda1_5_6_1_offset);
void ccsd_lambda1_5_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_lambda1_5_6_1,
                       Integer *k_lambda1_5_6_1_offset, Integer *d_lambda1_5_1,
                       Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_7_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda1_5_1, Integer *k_lambda1_5_1_offset);
void ccsd_lambda1_5_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda1_5_1, Integer *k_lambda1_5_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_6_1_(Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda1_6_1, Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_2_1_(Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_6_2_1,
                         Integer *k_lambda1_6_2_1_offset);
void ccsd_lambda1_6_2_2_1_(Integer *d_v, Integer *k_v_offset,
                           Integer *d_lambda1_6_2_2_1,
                           Integer *k_lambda1_6_2_2_1_offset);
void ccsd_lambda1_6_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_v, Integer *k_v_offset,
                           Integer *d_lambda1_6_2_2_1,
                           Integer *k_lambda1_6_2_2_1_offset);
void ccsd_lambda1_6_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                         Integer *d_lambda1_6_2_2_1,
                         Integer *k_lambda1_6_2_2_1_offset,
                         Integer *d_lambda1_6_2_1,
                         Integer *k_lambda1_6_2_1_offset);
void ccsd_lambda1_6_2_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                         Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_6_2_1,
                         Integer *k_lambda1_6_2_1_offset);
void ccsd_lambda1_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_6_2_1,
                       Integer *k_lambda1_6_2_1_offset, Integer *d_lambda1_6_1,
                       Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_3_1_(Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_6_3_1,
                         Integer *k_lambda1_6_3_1_offset);
void ccsd_lambda1_6_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_6_3_1,
                         Integer *k_lambda1_6_3_1_offset);
void ccsd_lambda1_6_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_6_3_1,
                       Integer *k_lambda1_6_3_1_offset, Integer *d_lambda1_6_1,
                       Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_4_1_(Integer *d_f, Integer *k_f_offset,
                         Integer *d_lambda1_6_4_1,
                         Integer *k_lambda1_6_4_1_offset);
void ccsd_lambda1_6_4_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_6_4_1,
                         Integer *k_lambda1_6_4_1_offset);
void ccsd_lambda1_6_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_lambda1_6_4_1,
                       Integer *k_lambda1_6_4_1_offset, Integer *d_lambda1_6_1,
                       Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_5_1_(Integer *d_v, Integer *k_v_offset,
                         Integer *d_lambda1_6_5_1,
                         Integer *k_lambda1_6_5_1_offset);
void ccsd_lambda1_6_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_lambda1_6_5_1,
                         Integer *k_lambda1_6_5_1_offset);
void ccsd_lambda1_6_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_lambda1_6_5_1,
                       Integer *k_lambda1_6_5_1_offset, Integer *d_lambda1_6_1,
                       Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda1_6_1, Integer *k_lambda1_6_1_offset);
void ccsd_lambda1_6_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda1_6_1, Integer *k_lambda1_6_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_7_1_(Integer *d_v, Integer *k_v_offset,
                       Integer *d_lambda1_7_1, Integer *k_lambda1_7_1_offset);
void ccsd_lambda1_7_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                       Integer *k_v_offset, Integer *d_lambda1_7_1,
                       Integer *k_lambda1_7_1_offset);
void ccsd_lambda1_7_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                     Integer *d_lambda1_7_1, Integer *k_lambda1_7_1_offset,
                     Integer *d_i0, Integer *k_i0_offset);
void ccsd_lambda1_8_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_8_1, Integer *k_lambda1_8_1_offset);
void ccsd_lambda1_8_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_y_ov, Integer *k_y_ov_offset,
                       Integer *d_lambda1_8_1, Integer *k_lambda1_8_1_offset);
void ccsd_lambda1_8_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                         Integer *d_y_ov, Integer *k_y_ov_offset,
                         Integer *d_lambda1_8_3_1,
                         Integer *k_lambda1_8_3_1_offset);
void ccsd_lambda1_8_3_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                         Integer *d_y_oovv, Integer *k_y_oovv_offset,
                         Integer *d_lambda1_8_3_1,
                         Integer *k_lambda1_8_3_1_offset);
void ccsd_lambda1_8_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                       Integer *d_lambda1_8_3_1,
                       Integer *k_lambda1_8_3_1_offset, Integer *d_lambda1_8_1,
                       Integer *k_lambda1_8_1_offset);
void ccsd_lambda1_8_4_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                         Integer *d_y_oovv, Integer *k_y_oovv_offset,
                         Integer *d_lambda1_8_4_1,
                         Integer *k_lambda1_8_4_1_offset);
void ccsd_lambda1_8_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_lambda1_8_4_1,
                       Integer *k_lambda1_8_4_1_offset, Integer *d_lambda1_8_1,
                       Integer *k_lambda1_8_1_offset);
void ccsd_lambda1_8_(Integer *d_lambda1_8_1, Integer *k_lambda1_8_1_offset,
                     Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                     Integer *k_i0_offset);
void ccsd_lambda1_9_1_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_y_ov,
                       Integer *k_y_ov_offset, Integer *d_lambda1_9_1,
                       Integer *k_lambda1_9_1_offset);
void ccsd_lambda1_9_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                       Integer *d_y_oovv, Integer *k_y_oovv_offset,
                       Integer *d_lambda1_9_1, Integer *k_lambda1_9_1_offset);
void ccsd_lambda1_9_(Integer *d_lambda1_9_1, Integer *k_lambda1_9_1_offset,
                     Integer *d_f, Integer *k_f_offset, Integer *d_i0,
                     Integer *k_i0_offset);
void ccsd_lambda1_10_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda1_10_1,
                        Integer *k_lambda1_10_1_offset);
void ccsd_lambda1_10_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1_10_1,
                        Integer *k_lambda1_10_1_offset);
void ccsd_lambda1_10_(Integer *d_lambda1_10_1, Integer *k_lambda1_10_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda1_11_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda1_11_1,
                        Integer *k_lambda1_11_1_offset);
void ccsd_lambda1_11_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1_11_1,
                        Integer *k_lambda1_11_1_offset);
void ccsd_lambda1_11_(Integer *d_lambda1_11_1, Integer *k_lambda1_11_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda1_12_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1_12_1,
                        Integer *k_lambda1_12_1_offset);
void ccsd_lambda1_12_(Integer *d_lambda1_12_1, Integer *k_lambda1_12_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda1_13_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda1_13_1,
                        Integer *k_lambda1_13_1_offset);
void ccsd_lambda1_13_2_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1_13_2_1,
                          Integer *k_lambda1_13_2_1_offset);
void ccsd_lambda1_13_2_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_y_oovv, Integer *k_y_oovv_offset,
                            Integer *d_lambda1_13_2_2_1,
                            Integer *k_lambda1_13_2_2_1_offset);
void ccsd_lambda1_13_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1_13_2_2_1,
                          Integer *k_lambda1_13_2_2_1_offset,
                          Integer *d_lambda1_13_2_1,
                          Integer *k_lambda1_13_2_1_offset);
void ccsd_lambda1_13_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda1_13_2_1,
                        Integer *k_lambda1_13_2_1_offset,
                        Integer *d_lambda1_13_1,
                        Integer *k_lambda1_13_1_offset);
void ccsd_lambda1_13_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1_13_3_1,
                          Integer *k_lambda1_13_3_1_offset);
void ccsd_lambda1_13_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_lambda1_13_3_1,
                        Integer *k_lambda1_13_3_1_offset,
                        Integer *d_lambda1_13_1,
                        Integer *k_lambda1_13_1_offset);
void ccsd_lambda1_13_4_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1_13_4_1,
                          Integer *k_lambda1_13_4_1_offset);
void ccsd_lambda1_13_4_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda1_13_4_1,
                        Integer *k_lambda1_13_4_1_offset,
                        Integer *d_lambda1_13_1,
                        Integer *k_lambda1_13_1_offset);
void ccsd_lambda1_13_(Integer *d_lambda1_13_1, Integer *k_lambda1_13_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda1_14_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1_14_1,
                        Integer *k_lambda1_14_1_offset);
void ccsd_lambda1_14_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1_14_2_1,
                          Integer *k_lambda1_14_2_1_offset);
void ccsd_lambda1_14_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda1_14_2_1,
                        Integer *k_lambda1_14_2_1_offset,
                        Integer *d_lambda1_14_1,
                        Integer *k_lambda1_14_1_offset);
void ccsd_lambda1_14_(Integer *d_lambda1_14_1, Integer *k_lambda1_14_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);
void ccsd_lambda1_15_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                        Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1_15_1,
                        Integer *k_lambda1_15_1_offset);
void ccsd_lambda1_15_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1_15_2_1,
                          Integer *k_lambda1_15_2_1_offset);
void ccsd_lambda1_15_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                        Integer *d_lambda1_15_2_1,
                        Integer *k_lambda1_15_2_1_offset,
                        Integer *d_lambda1_15_1,
                        Integer *k_lambda1_15_1_offset);
void ccsd_lambda1_15_(Integer *d_lambda1_15_1, Integer *k_lambda1_15_1_offset,
                      Integer *d_v, Integer *k_v_offset, Integer *d_i0,
                      Integer *k_i0_offset);

void offset_ccsd_lambda1_2_1_(Integer *l_lambda1_2_1_offset,
                              Integer *k_lambda1_2_1_offset,
                              Integer *size_lambda1_2_1);
void offset_ccsd_lambda1_2_2_1_(Integer *l_lambda1_2_2_1_offset,
                                Integer *k_lambda1_2_2_1_offset,
                                Integer *size_lambda1_2_2_1);
void offset_ccsd_lambda1_3_1_(Integer *l_lambda1_3_1_offset,
                              Integer *k_lambda1_3_1_offset,
                              Integer *size_lambda1_3_1);
void offset_ccsd_lambda1_3_3_1_(Integer *l_lambda1_3_3_1_offset,
                                Integer *k_lambda1_3_3_1_offset,
                                Integer *size_lambda1_3_3_1);
void offset_ccsd_lambda1_5_1_(Integer *l_lambda1_5_1_offset,
                              Integer *k_lambda1_5_1_offset,
                              Integer *size_lambda1_5_1);
void offset_ccsd_lambda1_5_2_1_(Integer *l_lambda1_5_2_1_offset,
                                Integer *k_lambda1_5_2_1_offset,
                                Integer *size_lambda1_5_2_1);
void offset_ccsd_lambda1_5_2_2_1_(Integer *l_lambda1_5_2_2_1_offset,
                                  Integer *k_lambda1_5_2_2_1_offset,
                                  Integer *size_lambda1_5_2_2_1);
void offset_ccsd_lambda1_5_3_1_(Integer *l_lambda1_5_3_1_offset,
                                Integer *k_lambda1_5_3_1_offset,
                                Integer *size_lambda1_5_3_1);
void offset_ccsd_lambda1_5_5_1_(Integer *l_lambda1_5_5_1_offset,
                                Integer *k_lambda1_5_5_1_offset,
                                Integer *size_lambda1_5_5_1);
void offset_ccsd_lambda1_5_6_1_(Integer *l_lambda1_5_6_1_offset,
                                Integer *k_lambda1_5_6_1_offset,
                                Integer *size_lambda1_5_6_1);
void offset_ccsd_lambda1_6_1_(Integer *l_lambda1_6_1_offset,
                              Integer *k_lambda1_6_1_offset,
                              Integer *size_lambda1_6_1);
void offset_ccsd_lambda1_6_2_1_(Integer *l_lambda1_6_2_1_offset,
                                Integer *k_lambda1_6_2_1_offset,
                                Integer *size_lambda1_6_2_1);
void offset_ccsd_lambda1_6_2_2_1_(Integer *l_lambda1_6_2_2_1_offset,
                                  Integer *k_lambda1_6_2_2_1_offset,
                                  Integer *size_lambda1_6_2_2_1);
void offset_ccsd_lambda1_6_3_1_(Integer *l_lambda1_6_3_1_offset,
                                Integer *k_lambda1_6_3_1_offset,
                                Integer *size_lambda1_6_3_1);
void offset_ccsd_lambda1_6_4_1_(Integer *l_lambda1_6_4_1_offset,
                                Integer *k_lambda1_6_4_1_offset,
                                Integer *size_lambda1_6_4_1);
void offset_ccsd_lambda1_6_5_1_(Integer *l_lambda1_6_5_1_offset,
                                Integer *k_lambda1_6_5_1_offset,
                                Integer *size_lambda1_6_5_1);
void offset_ccsd_lambda1_7_1_(Integer *l_lambda1_7_1_offset,
                              Integer *k_lambda1_7_1_offset,
                              Integer *size_lambda1_7_1);
void offset_ccsd_lambda1_8_1_(Integer *l_lambda1_8_1_offset,
                              Integer *k_lambda1_8_1_offset,
                              Integer *size_lambda1_8_1);
void offset_ccsd_lambda1_8_3_1_(Integer *l_lambda1_8_3_1_offset,
                                Integer *k_lambda1_8_3_1_offset,
                                Integer *size_lambda1_8_3_1);
void offset_ccsd_lambda1_8_4_1_(Integer *l_lambda1_8_4_1_offset,
                                Integer *k_lambda1_8_4_1_offset,
                                Integer *size_lambda1_8_4_1);
void offset_ccsd_lambda1_9_1_(Integer *l_lambda1_9_1_offset,
                              Integer *k_lambda1_9_1_offset,
                              Integer *size_lambda1_9_1);
void offset_ccsd_lambda1_10_1_(Integer *l_lambda1_10_1_offset,
                               Integer *k_lambda1_10_1_offset,
                               Integer *size_lambda1_10_1);
void offset_ccsd_lambda1_11_1_(Integer *l_lambda1_11_1_offset,
                               Integer *k_lambda1_11_1_offset,
                               Integer *size_lambda1_11_1);
void offset_ccsd_lambda1_12_1_(Integer *l_lambda1_12_1_offset,
                               Integer *k_lambda1_12_1_offset,
                               Integer *size_lambda1_12_1);
void offset_ccsd_lambda1_13_1_(Integer *l_lambda1_13_1_offset,
                               Integer *k_lambda1_13_1_offset,
                               Integer *size_lambda1_13_1);
void offset_ccsd_lambda1_13_2_1_(Integer *l_lambda1_13_2_1_offset,
                                 Integer *k_lambda1_13_2_1_offset,
                                 Integer *size_lambda1_13_2_1);
void offset_ccsd_lambda1_13_2_2_1_(Integer *l_lambda1_13_2_2_1_offset,
                                   Integer *k_lambda1_13_2_2_1_offset,
                                   Integer *size_lambda1_13_2_2_1);
void offset_ccsd_lambda1_13_3_1_(Integer *l_lambda1_13_3_1_offset,
                                 Integer *k_lambda1_13_3_1_offset,
                                 Integer *size_lambda1_13_3_1);
void offset_ccsd_lambda1_13_4_1_(Integer *l_lambda1_13_4_1_offset,
                                 Integer *k_lambda1_13_4_1_offset,
                                 Integer *size_lambda1_13_4_1);
void offset_ccsd_lambda1_14_1_(Integer *l_lambda1_14_1_offset,
                               Integer *k_lambda1_14_1_offset,
                               Integer *size_lambda1_14_1);
void offset_ccsd_lambda1_14_2_1_(Integer *l_lambda1_14_2_1_offset,
                                 Integer *k_lambda1_14_2_1_offset,
                                 Integer *size_lambda1_14_2_1);
void offset_ccsd_lambda1_15_1_(Integer *l_lambda1_15_1_offset,
                               Integer *k_lambda1_15_1_offset,
                               Integer *size_lambda1_15_1);
void offset_ccsd_lambda1_15_2_1_(Integer *l_lambda1_15_2_1_offset,
                                 Integer *k_lambda1_15_2_1_offset,
                                 Integer *size_lambda1_15_2_1);
}

namespace tamm {

extern "C" {
#if 1
void ccsd_lambda1_cxx_(Integer *d_f, Integer *d_i0, Integer *d_t_vo,
                       Integer *d_t_vvoo, Integer *d_v, Integer *d_y_ov,
                       Integer *d_y_oovv, Integer *k_f_offset,
                       Integer *k_i0_offset, Integer *k_t_vo_offset,
                       Integer *k_t_vvoo_offset, Integer *k_v_offset,
                       Integer *k_y_ov_offset, Integer *k_y_oovv_offset) {
#else
void ccsd_lambda1_cxx_(Integer *d_t_vvoo, Integer *d_f, Integer *d_i0,
                       Integer *d_y_ov, Integer *d_y_oovv, Integer *d_t_vo,
                       Integer *d_v, Integer *k_t_vvoo_offset,
                       Integer *k_f_offset, Integer *k_i0_offset,
                       Integer *k_y_ov_offset, Integer *k_y_oovv_offset,
                       Integer *k_t_vo_offset, Integer *k_v_offset) {
#endif  // if 1

  static bool set_lambda1 = true;

  Assignment op_lambda1_1;
  Assignment op_lambda1_2_1;
  Assignment op_lambda1_2_2_1;
  Assignment op_lambda1_3_1;
  Assignment op_lambda1_5_1;
  Assignment op_lambda1_5_2_1;
  Assignment op_lambda1_5_2_2_1;
  Assignment op_lambda1_5_3_1;
  Assignment op_lambda1_5_5_1;
  Assignment op_lambda1_5_6_1;
  Assignment op_lambda1_6_1;
  Assignment op_lambda1_6_2_1;
  Assignment op_lambda1_6_2_2_1;
  Assignment op_lambda1_6_3_1;
  Assignment op_lambda1_6_4_1;
  Assignment op_lambda1_6_5_1;
  Assignment op_lambda1_7_1;
  Assignment op_lambda1_8_1;
  Multiplication op_lambda1_2_2_2;
  Multiplication op_lambda1_2_2;
  Multiplication op_lambda1_2_3;
  Multiplication op_lambda1_2_4;
  Multiplication op_lambda1_2;
  Multiplication op_lambda1_3_2;
  Multiplication op_lambda1_3_3_1;
  Multiplication op_lambda1_3_3;
  Multiplication op_lambda1_3;
  Multiplication op_lambda1_4;
  Multiplication op_lambda1_5_2_2_2;
  Multiplication op_lambda1_5_2_2;
  Multiplication op_lambda1_5_2_3;
  Multiplication op_lambda1_5_2_4;
  Multiplication op_lambda1_5_2;
  Multiplication op_lambda1_5_3_2;
  Multiplication op_lambda1_5_3;
  Multiplication op_lambda1_5_4;
  Multiplication op_lambda1_5_5_2;
  Multiplication op_lambda1_5_5;
  Multiplication op_lambda1_5_6_2;
  Multiplication op_lambda1_5_6;
  Multiplication op_lambda1_5_7;
  Multiplication op_lambda1_5;
  Multiplication op_lambda1_6_2_2_2;
  Multiplication op_lambda1_6_2_2;
  Multiplication op_lambda1_6_2_3;
  Multiplication op_lambda1_6_2;
  Multiplication op_lambda1_6_3_2;
  Multiplication op_lambda1_6_3;
  Multiplication op_lambda1_6_4_2;
  Multiplication op_lambda1_6_4;
  Multiplication op_lambda1_6_5_2;
  Multiplication op_lambda1_6_5;
  Multiplication op_lambda1_6_6;
  Multiplication op_lambda1_6;
  Multiplication op_lambda1_7_2;
  Multiplication op_lambda1_7;
  Multiplication op_lambda1_8_2;
  Multiplication op_lambda1_8_3_1;
  Multiplication op_lambda1_8_3_2;
  Multiplication op_lambda1_8_3;
  Multiplication op_lambda1_8_4_1;
  Multiplication op_lambda1_8_4;
  Multiplication op_lambda1_8;
  Multiplication op_lambda1_9_1;
  Multiplication op_lambda1_9_2;
  Multiplication op_lambda1_9;
  Multiplication op_lambda1_10_1;
  Multiplication op_lambda1_10_2;
  Multiplication op_lambda1_10;
  Multiplication op_lambda1_11_1;
  Multiplication op_lambda1_11_2;
  Multiplication op_lambda1_11;
  Multiplication op_lambda1_12_1;
  Multiplication op_lambda1_12;
  Multiplication op_lambda1_13_1;
  Multiplication op_lambda1_13_2_1;
  Multiplication op_lambda1_13_2_2_1;
  Multiplication op_lambda1_13_2_2;
  Multiplication op_lambda1_13_2;
  Multiplication op_lambda1_13_3_1;
  Multiplication op_lambda1_13_3;
  Multiplication op_lambda1_13_4_1;
  Multiplication op_lambda1_13_4;
  Multiplication op_lambda1_13;
  Multiplication op_lambda1_14_1;
  Multiplication op_lambda1_14_2_1;
  Multiplication op_lambda1_14_2;
  Multiplication op_lambda1_14;
  Multiplication op_lambda1_15_1;
  Multiplication op_lambda1_15_2_1;
  Multiplication op_lambda1_15_2;
  Multiplication op_lambda1_15;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_lambda1) {
    ccsd_lambda1_equations(&eqs);
    set_lambda1 = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *f = &tensors["f"];
  Tensor *y_ov = &tensors["y_ov"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *v = &tensors["v"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *y_oovv = &tensors["y_oovv"];
  Tensor *lambda1_15_2_1 = &tensors["lambda1_15_2_1"];
  Tensor *lambda1_3_3_1 = &tensors["lambda1_3_3_1"];
  Tensor *lambda1_5_5_1 = &tensors["lambda1_5_5_1"];
  Tensor *lambda1_5_3_1 = &tensors["lambda1_5_3_1"];
  Tensor *lambda1_5_2_1 = &tensors["lambda1_5_2_1"];
  Tensor *lambda1_6_4_1 = &tensors["lambda1_6_4_1"];
  Tensor *lambda1_6_2_2_1 = &tensors["lambda1_6_2_2_1"];
  Tensor *lambda1_3_1 = &tensors["lambda1_3_1"];
  Tensor *lambda1_2_1 = &tensors["lambda1_2_1"];
  Tensor *lambda1_6_2_1 = &tensors["lambda1_6_2_1"];
  Tensor *lambda1_10_1 = &tensors["lambda1_10_1"];
  Tensor *lambda1_6_3_1 = &tensors["lambda1_6_3_1"];
  Tensor *lambda1_8_1 = &tensors["lambda1_8_1"];
  Tensor *lambda1_7_1 = &tensors["lambda1_7_1"];
  Tensor *lambda1_5_2_2_1 = &tensors["lambda1_5_2_2_1"];
  Tensor *lambda1_6_1 = &tensors["lambda1_6_1"];
  Tensor *lambda1_6_5_1 = &tensors["lambda1_6_5_1"];
  Tensor *lambda1_13_2_2_1 = &tensors["lambda1_13_2_2_1"];
  Tensor *lambda1_11_1 = &tensors["lambda1_11_1"];
  Tensor *lambda1_5_1 = &tensors["lambda1_5_1"];
  Tensor *lambda1_8_4_1 = &tensors["lambda1_8_4_1"];
  Tensor *lambda1_13_2_1 = &tensors["lambda1_13_2_1"];
  Tensor *lambda1_14_1 = &tensors["lambda1_14_1"];
  Tensor *lambda1_13_3_1 = &tensors["lambda1_13_3_1"];
  Tensor *lambda1_15_1 = &tensors["lambda1_15_1"];
  Tensor *lambda1_8_3_1 = &tensors["lambda1_8_3_1"];
  Tensor *lambda1_14_2_1 = &tensors["lambda1_14_2_1"];
  Tensor *lambda1_12_1 = &tensors["lambda1_12_1"];
  Tensor *lambda1_13_4_1 = &tensors["lambda1_13_4_1"];
  Tensor *lambda1_5_6_1 = &tensors["lambda1_5_6_1"];
  Tensor *lambda1_9_1 = &tensors["lambda1_9_1"];
  Tensor *lambda1_2_2_1 = &tensors["lambda1_2_2_1"];
  Tensor *lambda1_13_1 = &tensors["lambda1_13_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  // t_vo->set_dist(dist_nwma);
  // t_vo->set_dist(dist_nw);
  // t_vvoo->set_dist(dist_nw);
  // y_oovv->set_dist(dist_nwi);
  // lambda1_11_1->set_dist(dist_nw);
  // lambda1_6_2_2_1->set_dist(dist_nw);
  // lambda1_6_2_1->set_dist(dist_nwi);
  // lambda1_6_3_1->set_dist(dist_nw);
  // lambda1_6_5_1->set_dist(dist_nw);
  // y_oovv->set_dist(dist_nwi);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);
  t_vo->attach(*k_t_vo_offset, 0, *d_t_vo);
  t_vvoo->attach(*k_t_vvoo_offset, 0, *d_t_vvoo);
  y_ov->attach(*k_y_ov_offset, 0, *d_y_ov);
  y_oovv->attach(*k_y_oovv_offset, 0, *d_y_oovv);

  // lambda1_6_3_1->set_irrep(Variables::irrep_v());
  y_ov->set_irrep(Variables::irrep_y());
  y_oovv->set_irrep(Variables::irrep_y());
  lambda1_11_1->set_irrep(Variables::irrep_y());
  i0->set_irrep(Variables::irrep_y());

#if 1
  schedule_linear(&tensors, &ops);
  // schedule_linear_lazy(tensors, &ops);
  //  schedule_levels(tensors, &ops);
#else
  op_lambda1_1 = ops[0].add;
  op_lambda1_2_1 = ops[1].add;
  op_lambda1_2_2_1 = ops[2].add;
  op_lambda1_2_2_2 = ops[3].mult;
  op_lambda1_2_2 = ops[4].mult;
  op_lambda1_2_3 = ops[5].mult;
  op_lambda1_2_4 = ops[6].mult;
  op_lambda1_2 = ops[7].mult;
  op_lambda1_3_1 = ops[8].add;
  op_lambda1_3_2 = ops[9].mult;
  op_lambda1_3_3_1 = ops[10].mult;
  op_lambda1_3_3 = ops[11].mult;
  op_lambda1_3 = ops[12].mult;
  op_lambda1_4 = ops[13].mult;
  op_lambda1_5_1 = ops[14].add;
  op_lambda1_5_2_1 = ops[15].add;
  op_lambda1_5_2_2_1 = ops[16].add;
  op_lambda1_5_2_2_2 = ops[17].mult;
  op_lambda1_5_2_2 = ops[18].mult;
  op_lambda1_5_2_3 = ops[19].mult;
  op_lambda1_5_2_4 = ops[20].mult;
  op_lambda1_5_2 = ops[21].mult;
  op_lambda1_5_3_1 = ops[22].add;
  op_lambda1_5_3_2 = ops[23].mult;
  op_lambda1_5_3 = ops[24].mult;
  op_lambda1_5_4 = ops[25].mult;
  op_lambda1_5_5_1 = ops[26].add;
  op_lambda1_5_5_2 = ops[27].mult;
  op_lambda1_5_5 = ops[28].mult;
  op_lambda1_5_6_1 = ops[29].add;
  op_lambda1_5_6_2 = ops[30].mult;
  op_lambda1_5_6 = ops[31].mult;
  op_lambda1_5_7 = ops[32].mult;
  op_lambda1_5 = ops[33].mult;
  op_lambda1_6_1 = ops[34].add;
  op_lambda1_6_2_1 = ops[35].add;
  op_lambda1_6_2_2_1 = ops[36].add;
  op_lambda1_6_2_2_2 = ops[37].mult;
  op_lambda1_6_2_2 = ops[38].mult;
  op_lambda1_6_2_3 = ops[39].mult;
  op_lambda1_6_2 = ops[40].mult;
  op_lambda1_6_3_1 = ops[41].add;
  op_lambda1_6_3_2 = ops[42].mult;
  op_lambda1_6_3 = ops[43].mult;
  op_lambda1_6_4_1 = ops[44].add;
  op_lambda1_6_4_2 = ops[45].mult;
  op_lambda1_6_4 = ops[46].mult;
  op_lambda1_6_5_1 = ops[47].add;
  op_lambda1_6_5_2 = ops[48].mult;
  op_lambda1_6_5 = ops[49].mult;
  op_lambda1_6_6 = ops[50].mult;
  op_lambda1_6 = ops[51].mult;
  op_lambda1_7_1 = ops[52].add;
  op_lambda1_7_2 = ops[53].mult;
  op_lambda1_7 = ops[54].mult;
  op_lambda1_8_1 = ops[55].add;
  op_lambda1_8_2 = ops[56].mult;
  op_lambda1_8_3_1 = ops[57].mult;
  op_lambda1_8_3_2 = ops[58].mult;
  op_lambda1_8_3 = ops[59].mult;
  op_lambda1_8_4_1 = ops[60].mult;
  op_lambda1_8_4 = ops[61].mult;
  op_lambda1_8 = ops[62].mult;
  op_lambda1_9_1 = ops[63].mult;
  op_lambda1_9_2 = ops[64].mult;
  op_lambda1_9 = ops[65].mult;
  op_lambda1_10_1 = ops[66].mult;
  op_lambda1_10_2 = ops[67].mult;
  op_lambda1_10 = ops[68].mult;
  op_lambda1_11_1 = ops[69].mult;
  op_lambda1_11_2 = ops[70].mult;
  op_lambda1_11 = ops[71].mult;
  op_lambda1_12_1 = ops[72].mult;
  op_lambda1_12 = ops[73].mult;
  op_lambda1_13_1 = ops[74].mult;
  op_lambda1_13_2_1 = ops[75].mult;
  op_lambda1_13_2_2_1 = ops[76].mult;
  op_lambda1_13_2_2 = ops[77].mult;
  op_lambda1_13_2 = ops[78].mult;
  op_lambda1_13_3_1 = ops[79].mult;
  op_lambda1_13_3 = ops[80].mult;
  op_lambda1_13_4_1 = ops[81].mult;
  op_lambda1_13_4 = ops[82].mult;
  op_lambda1_13 = ops[83].mult;
  op_lambda1_14_1 = ops[84].mult;
  op_lambda1_14_2_1 = ops[85].mult;
  op_lambda1_14_2 = ops[86].mult;
  op_lambda1_14 = ops[87].mult;
  op_lambda1_15_1 = ops[88].mult;
  op_lambda1_15_2_1 = ops[89].mult;
  op_lambda1_15_2 = ops[90].mult;
  op_lambda1_15 = ops[91].mult;

  CorFortran(1, &op_lambda1_1, ccsd_lambda1_1_);
  CorFortran(1, lambda1_2_1, offset_ccsd_lambda1_2_1_);
  CorFortran(1, &op_lambda1_2_1, ccsd_lambda1_2_1_);
  CorFortran(1, lambda1_2_2_1, offset_ccsd_lambda1_2_2_1_);
  CorFortran(1, &op_lambda1_2_2_1, ccsd_lambda1_2_2_1_);
  CorFortran(1, &op_lambda1_2_2_2, ccsd_lambda1_2_2_2_);
  CorFortran(1, &op_lambda1_2_2, ccsd_lambda1_2_2_);
  destroy(lambda1_2_2_1);
  CorFortran(1, &op_lambda1_2_3, ccsd_lambda1_2_3_);
  CorFortran(1, &op_lambda1_2_4, ccsd_lambda1_2_4_);
  CorFortran(1, &op_lambda1_2, ccsd_lambda1_2_);
  destroy(lambda1_2_1);
  CorFortran(1, lambda1_3_1, offset_ccsd_lambda1_3_1_);
  CorFortran(1, &op_lambda1_3_1, ccsd_lambda1_3_1_);
  CorFortran(1, &op_lambda1_3_2, ccsd_lambda1_3_2_);
  CorFortran(1, lambda1_3_3_1, offset_ccsd_lambda1_3_3_1_);
  CorFortran(1, &op_lambda1_3_3_1, ccsd_lambda1_3_3_1_);
  CorFortran(1, &op_lambda1_3_3, ccsd_lambda1_3_3_);
  destroy(lambda1_3_3_1);
  CorFortran(1, &op_lambda1_3, ccsd_lambda1_3_);
  destroy(lambda1_3_1);
  CorFortran(1, &op_lambda1_4, ccsd_lambda1_4_);
  CorFortran(1, lambda1_5_1, offset_ccsd_lambda1_5_1_);
  CorFortran(1, &op_lambda1_5_1, ccsd_lambda1_5_1_);
  CorFortran(1, lambda1_5_2_1, offset_ccsd_lambda1_5_2_1_);
  CorFortran(1, &op_lambda1_5_2_1, ccsd_lambda1_5_2_1_);
  CorFortran(1, lambda1_5_2_2_1, offset_ccsd_lambda1_5_2_2_1_);
  CorFortran(1, &op_lambda1_5_2_2_1, ccsd_lambda1_5_2_2_1_);
  CorFortran(1, &op_lambda1_5_2_2_2, ccsd_lambda1_5_2_2_2_);
  CorFortran(1, &op_lambda1_5_2_2, ccsd_lambda1_5_2_2_);
  destroy(lambda1_5_2_2_1);
  CorFortran(1, &op_lambda1_5_2_3, ccsd_lambda1_5_2_3_);
  CorFortran(1, &op_lambda1_5_2_4, ccsd_lambda1_5_2_4_);
  CorFortran(1, &op_lambda1_5_2, ccsd_lambda1_5_2_);
  destroy(lambda1_5_2_1);
  CorFortran(1, lambda1_5_3_1, offset_ccsd_lambda1_5_3_1_);
  CorFortran(1, &op_lambda1_5_3_1, ccsd_lambda1_5_3_1_);
  CorFortran(1, &op_lambda1_5_3_2, ccsd_lambda1_5_3_2_);
  CorFortran(1, &op_lambda1_5_3, ccsd_lambda1_5_3_);
  destroy(lambda1_5_3_1);
  CorFortran(1, &op_lambda1_5_4, ccsd_lambda1_5_4_);
  CorFortran(1, lambda1_5_5_1, offset_ccsd_lambda1_5_5_1_);
  CorFortran(1, &op_lambda1_5_5_1, ccsd_lambda1_5_5_1_);
  CorFortran(1, &op_lambda1_5_5_2, ccsd_lambda1_5_5_2_);
  CorFortran(1, &op_lambda1_5_5, ccsd_lambda1_5_5_);
  destroy(lambda1_5_5_1);
  CorFortran(1, lambda1_5_6_1, offset_ccsd_lambda1_5_6_1_);
  CorFortran(1, &op_lambda1_5_6_1, ccsd_lambda1_5_6_1_);
  CorFortran(1, &op_lambda1_5_6_2, ccsd_lambda1_5_6_2_);
  CorFortran(1, &op_lambda1_5_6, ccsd_lambda1_5_6_);
  destroy(lambda1_5_6_1);
  CorFortran(1, &op_lambda1_5_7, ccsd_lambda1_5_7_);
  CorFortran(1, &op_lambda1_5, ccsd_lambda1_5_);
  destroy(lambda1_5_1);
  CorFortran(1, lambda1_6_1, offset_ccsd_lambda1_6_1_);
  CorFortran(1, &op_lambda1_6_1, ccsd_lambda1_6_1_);
#if 1  // following block works entirely in fortran or c++,
  CorFortran(1, lambda1_6_2_1, offset_ccsd_lambda1_6_2_1_);
  CorFortran(1, &op_lambda1_6_2_1, ccsd_lambda1_6_2_1_);
  CorFortran(1, lambda1_6_2_2_1, offset_ccsd_lambda1_6_2_2_1_);
  CorFortran(1, &op_lambda1_6_2_2_1, ccsd_lambda1_6_2_2_1_);
  CorFortran(1, &op_lambda1_6_2_2_2, ccsd_lambda1_6_2_2_2_);
  CorFortran(1, &op_lambda1_6_2_2, ccsd_lambda1_6_2_2_); // solved by replacing the multiplier from -2 to -1
  destroy(lambda1_6_2_2_1);
  CorFortran(1, &op_lambda1_6_2_3, ccsd_lambda1_6_2_3_);
  CorFortran(1, &op_lambda1_6_2, ccsd_lambda1_6_2_);
#else
  CorFortran(0, lambda1_6_2_1, offset_ccsd_lambda1_6_2_1_);
  CorFortran(0, &op_lambda1_6_2_1, ccsd_lambda1_6_2_1_);
  CorFortran(0, lambda1_6_2_2_1, offset_ccsd_lambda1_6_2_2_1_);
  CorFortran(0, &op_lambda1_6_2_2_1, ccsd_lambda1_6_2_2_1_);
  CorFortran(0, &op_lambda1_6_2_2_2, ccsd_lambda1_6_2_2_2_);
  CorFortran(0, &op_lambda1_6_2_2, ccsd_lambda1_6_2_2_);
  destroy(lambda1_6_2_2_1);
  CorFortran(0, &op_lambda1_6_2_3, ccsd_lambda1_6_2_3_);
  CorFortran(0, &op_lambda1_6_2, ccsd_lambda1_6_2_);
#endif  // if 1 or 0 etc
  destroy(lambda1_6_2_1);
  CorFortran(1, lambda1_6_3_1, offset_ccsd_lambda1_6_3_1_);
  CorFortran(1, &op_lambda1_6_3_1, ccsd_lambda1_6_3_1_);
  CorFortran(1, &op_lambda1_6_3_2, ccsd_lambda1_6_3_2_);
  CorFortran(1, &op_lambda1_6_3, ccsd_lambda1_6_3_);  // solved by replacing the multiplier from -2 to -1
  destroy(lambda1_6_3_1);
  CorFortran(1, lambda1_6_4_1, offset_ccsd_lambda1_6_4_1_);
  CorFortran(1, &op_lambda1_6_4_1, ccsd_lambda1_6_4_1_);
  CorFortran(1, &op_lambda1_6_4_2, ccsd_lambda1_6_4_2_);
  CorFortran(1, &op_lambda1_6_4, ccsd_lambda1_6_4_);
  destroy(lambda1_6_4_1);
#if 1  // following block works entirely in fortran or c++
  CorFortran(1, lambda1_6_5_1, offset_ccsd_lambda1_6_5_1_);
  CorFortran(1, &op_lambda1_6_5_1, ccsd_lambda1_6_5_1_);
  CorFortran(1, &op_lambda1_6_5_2, ccsd_lambda1_6_5_2_);
  CorFortran(1, &op_lambda1_6_5, ccsd_lambda1_6_5_);  // solved  by replacing the multiplier from -2 to -1
#else
  CorFortran(0, lambda1_6_5_1, offset_ccsd_lambda1_6_5_1_);
  CorFortran(0, &op_lambda1_6_5_1, ccsd_lambda1_6_5_1_);
  CorFortran(0, &op_lambda1_6_5_2, ccsd_lambda1_6_5_2_);
  CorFortran(0, &op_lambda1_6_5, ccsd_lambda1_6_5_);
#endif  // if 1 or 0 etc
  destroy(lambda1_6_5_1);
  CorFortran(1, &op_lambda1_6_6, ccsd_lambda1_6_6_);
  CorFortran(1, &op_lambda1_6, ccsd_lambda1_6_);
  destroy(lambda1_6_1);
  CorFortran(1, lambda1_7_1, offset_ccsd_lambda1_7_1_);
  CorFortran(1, &op_lambda1_7_1, ccsd_lambda1_7_1_);
  CorFortran(1, &op_lambda1_7_2, ccsd_lambda1_7_2_);
  CorFortran(1, &op_lambda1_7, ccsd_lambda1_7_);
  destroy(lambda1_7_1);
  CorFortran(1, lambda1_8_1, offset_ccsd_lambda1_8_1_);
  CorFortran(1, &op_lambda1_8_1, ccsd_lambda1_8_1_);
  CorFortran(1, &op_lambda1_8_2, ccsd_lambda1_8_2_);
  CorFortran(1, lambda1_8_3_1, offset_ccsd_lambda1_8_3_1_);
  CorFortran(1, &op_lambda1_8_3_1, ccsd_lambda1_8_3_1_);
  CorFortran(1, &op_lambda1_8_3_2, ccsd_lambda1_8_3_2_);
  CorFortran(1, &op_lambda1_8_3, ccsd_lambda1_8_3_);
  destroy(lambda1_8_3_1);
  CorFortran(1, lambda1_8_4_1, offset_ccsd_lambda1_8_4_1_);
  CorFortran(1, &op_lambda1_8_4_1, ccsd_lambda1_8_4_1_);
  CorFortran(1, &op_lambda1_8_4, ccsd_lambda1_8_4_);
  destroy(lambda1_8_4_1);
  CorFortran(1, &op_lambda1_8, ccsd_lambda1_8_);
  destroy(lambda1_8_1);
  CorFortran(1, lambda1_9_1, offset_ccsd_lambda1_9_1_);
  CorFortran(1, &op_lambda1_9_1, ccsd_lambda1_9_1_);
  CorFortran(1, &op_lambda1_9_2, ccsd_lambda1_9_2_);
  CorFortran(1, &op_lambda1_9, ccsd_lambda1_9_);
  destroy(lambda1_9_1);
  CorFortran(1, lambda1_10_1, offset_ccsd_lambda1_10_1_);
  CorFortran(1, &op_lambda1_10_1, ccsd_lambda1_10_1_);
  CorFortran(1, &op_lambda1_10_2, ccsd_lambda1_10_2_);
  CorFortran(1, &op_lambda1_10, ccsd_lambda1_10_);
  destroy(lambda1_10_1);
  CorFortran(1, lambda1_11_1, offset_ccsd_lambda1_11_1_);
  CorFortran(1, &op_lambda1_11_1, ccsd_lambda1_11_1_);
  CorFortran(1, &op_lambda1_11_2, ccsd_lambda1_11_2_);
  CorFortran(1, &op_lambda1_11, ccsd_lambda1_11_);  // ok after bug fix
  destroy(lambda1_11_1);
#if 1  // following block works entirely in fortran or c++, ok after bug fix
  CorFortran(1, lambda1_12_1, offset_ccsd_lambda1_12_1_);
  CorFortran(1, &op_lambda1_12_1, ccsd_lambda1_12_1_);
  CorFortran(1, &op_lambda1_12, ccsd_lambda1_12_);
#else
  CorFortran(0, lambda1_12_1, offset_ccsd_lambda1_12_1_);
  CorFortran(0, &op_lambda1_12_1, ccsd_lambda1_12_1_);
  CorFortran(0, &op_lambda1_12, ccsd_lambda1_12_);
#endif  // if 1 or 0 etc
  destroy(lambda1_12_1);
#if 1  // following block works entirely in fortran or c++, ok after bug fix
  CorFortran(1, lambda1_13_1, offset_ccsd_lambda1_13_1_);
  CorFortran(1, &op_lambda1_13_1, ccsd_lambda1_13_1_);
  CorFortran(1, lambda1_13_2_1, offset_ccsd_lambda1_13_2_1_);
  CorFortran(1, &op_lambda1_13_2_1, ccsd_lambda1_13_2_1_);
  CorFortran(1, lambda1_13_2_2_1, offset_ccsd_lambda1_13_2_2_1_);
  CorFortran(1, &op_lambda1_13_2_2_1, ccsd_lambda1_13_2_2_1_);
  CorFortran(1, &op_lambda1_13_2_2, ccsd_lambda1_13_2_2_); // solved  by replacing the multiplier from 2 to 1
  destroy(lambda1_13_2_2_1);
  CorFortran(1, &op_lambda1_13_2, ccsd_lambda1_13_2_);
  destroy(lambda1_13_2_1);
  CorFortran(1, lambda1_13_3_1, offset_ccsd_lambda1_13_3_1_);
  CorFortran(1, &op_lambda1_13_3_1, ccsd_lambda1_13_3_1_);
  CorFortran(1, &op_lambda1_13_3, ccsd_lambda1_13_3_); // solved  by replacing the multiplier from 2 to 1
  destroy(lambda1_13_3_1);
  CorFortran(1, lambda1_13_4_1, offset_ccsd_lambda1_13_4_1_);
  CorFortran(1, &op_lambda1_13_4_1, ccsd_lambda1_13_4_1_);
  CorFortran(1, &op_lambda1_13_4, ccsd_lambda1_13_4_); // solved  by replacing the multiplier from -1 to -1/2
  destroy(lambda1_13_4_1);
  CorFortran(1, &op_lambda1_13, ccsd_lambda1_13_);
#else
  CorFortran(0, lambda1_13_1, offset_ccsd_lambda1_13_1_);
  CorFortran(0, &op_lambda1_13_1, ccsd_lambda1_13_1_);
  CorFortran(0, lambda1_13_2_1, offset_ccsd_lambda1_13_2_1_);
  CorFortran(0, &op_lambda1_13_2_1, ccsd_lambda1_13_2_1_);
  CorFortran(0, lambda1_13_2_2_1, offset_ccsd_lambda1_13_2_2_1_);
  CorFortran(0, &op_lambda1_13_2_2_1, ccsd_lambda1_13_2_2_1_);
  CorFortran(0, &op_lambda1_13_2_2, ccsd_lambda1_13_2_2_);
  destroy(lambda1_13_2_2_1);
  CorFortran(0, &op_lambda1_13_2, ccsd_lambda1_13_2_);
  destroy(lambda1_13_2_1);
  CorFortran(0, lambda1_13_3_1, offset_ccsd_lambda1_13_3_1_);
  CorFortran(0, &op_lambda1_13_3_1, ccsd_lambda1_13_3_1_);
  CorFortran(0, &op_lambda1_13_3, ccsd_lambda1_13_3_);
  destroy(lambda1_13_3_1);
  CorFortran(0, lambda1_13_4_1, offset_ccsd_lambda1_13_4_1_);
  CorFortran(0, &op_lambda1_13_4_1, ccsd_lambda1_13_4_1_);
  CorFortran(0, &op_lambda1_13_4, ccsd_lambda1_13_4_);
  destroy(lambda1_13_4_1);
  CorFortran(0, &op_lambda1_13, ccsd_lambda1_13_);
#endif  // if 1 or 0 etc
  destroy(lambda1_13_1);
#if 1  // following block works entirely in fortran or c++, after bug fix
  CorFortran(1, lambda1_14_1, offset_ccsd_lambda1_14_1_);
  CorFortran(1, &op_lambda1_14_1, ccsd_lambda1_14_1_);
  CorFortran(1, lambda1_14_2_1, offset_ccsd_lambda1_14_2_1_);
  CorFortran(1, &op_lambda1_14_2_1, ccsd_lambda1_14_2_1_);
  CorFortran(1, &op_lambda1_14_2, ccsd_lambda1_14_2_); // solved  by replacing the multiplier from 2 to 1
  destroy(lambda1_14_2_1);
  CorFortran(1, &op_lambda1_14, ccsd_lambda1_14_);
#else
  CorFortran(0, lambda1_14_1, offset_ccsd_lambda1_14_1_);
  CorFortran(0, &op_lambda1_14_1, ccsd_lambda1_14_1_);
  CorFortran(0, lambda1_14_2_1, offset_ccsd_lambda1_14_2_1_);
  CorFortran(0, &op_lambda1_14_2_1, ccsd_lambda1_14_2_1_);
  CorFortran(0, &op_lambda1_14_2, ccsd_lambda1_14_2_);
  destroy(lambda1_14_2_1);
  CorFortran(0, &op_lambda1_14, ccsd_lambda1_14_);
#endif  // if 1 or 0 etc
  destroy(lambda1_14_1);
#if 1  // following block works entirely in fortran or c++, after bug fix
  CorFortran(1, lambda1_15_1, offset_ccsd_lambda1_15_1_);
  CorFortran(1, &op_lambda1_15_1, ccsd_lambda1_15_1_);
  CorFortran(1, lambda1_15_2_1, offset_ccsd_lambda1_15_2_1_);
  CorFortran(1, &op_lambda1_15_2_1, ccsd_lambda1_15_2_1_);
  CorFortran(1, &op_lambda1_15_2, ccsd_lambda1_15_2_);
  destroy(lambda1_15_2_1);
  CorFortran(1, &op_lambda1_15, ccsd_lambda1_15_);
#else
  CorFortran(0, lambda1_15_1, offset_ccsd_lambda1_15_1_);
  CorFortran(0, &op_lambda1_15_1, ccsd_lambda1_15_1_);
  CorFortran(0, lambda1_15_2_1, offset_ccsd_lambda1_15_2_1_);
  CorFortran(0, &op_lambda1_15_2_1, ccsd_lambda1_15_2_1_);
  CorFortran(0, &op_lambda1_15_2, ccsd_lambda1_15_2_);
  destroy(lambda1_15_2_1);
  CorFortran(0, &op_lambda1_15, ccsd_lambda1_15_);
#endif  // if 1 or 0 etc
  destroy(lambda1_15_1);
#endif  // if 1 or 0 etc

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
  t_vo->detach();
  t_vvoo->detach();
  y_ov->detach();
  y_oovv->detach();
}
}  // extern C
};  // namespace tamm

