//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <vector>
#include <cmath>

// Define constants
const double s_g = (double)(1 << 24);
const int MAX_INT32 = ((1LL << 31) - 1);
const int MIN_INT32 = -(1LL << 31);
const int MAX_INT16 = ((1 << 16) - 1);
const int MIN_INT16 = -(1 << 16);
const int FLOAT_DEC_POINT = 24;

// Define helper functions
bool is_negative(long long x) {
    return x < 0;
}

int srs_int(long long x, unsigned char shft) {
    bool x_is_negative = is_negative(x);
    long long zerop5 = (shft == 0) ? 0 : (1 << (shft - 1));
    x = (zerop5 + (x_is_negative ? -x : x)) >> shft;
    x = x_is_negative ? -x : x;
    return ((x < MIN_INT32) ? MIN_INT32 : (x > MAX_INT32) ? MAX_INT32 : static_cast<int>(x));
}

short srs_short(long long x, unsigned char shft) {
    bool x_is_negative = is_negative(x);
    long long zerop5 = (shft == 0) ? 0 : (1 << (shft - 1));
    x = (zerop5 + (x_is_negative ? -x : x)) >> shft;
    x = x_is_negative ? -x : x;
    return ((x < MIN_INT16) ? MIN_INT16 : (x > MAX_INT16) ? MAX_INT16 : static_cast<short>(x));
}

int round_int(double x) {
    bool x_is_negative = is_negative(x);
    x = 0.5 + (x_is_negative ? -x : x);
    x = x_is_negative ? -x : x;
    return ((x < MIN_INT32) ? MIN_INT32 : (x > MAX_INT32) ? MAX_INT32 : static_cast<int>(x));
}

short round_short(double x) {
    bool x_is_negative = is_negative(x);
    x = 0.5 + (x_is_negative ? -x : x);
    x = x_is_negative ? -x : x;
    return ((x < MIN_INT16) ? MIN_INT16 : (x > MAX_INT16) ? MAX_INT16 : static_cast<short>(x));
}

// Define vector-based versions of functions
void vmul_x(int ite, const std::vector<std::vector<short>>& a, const std::vector<std::vector<short>>& b, unsigned char shft, std::vector<int>& y,
	int input_len, int hidden_len) {
    for (int i = 0; i < hidden_len * 4; i++) {
        long long sum = 0;
        for (int j = 0; j < input_len; j++) {
            sum += static_cast<long long>(a[ite][j]) * static_cast<long long>(b[j][i]);
        }
        y[i] = srs_int(sum, shft);
    }
}

void vmul_h(const std::vector<short>& a, const std::vector<std::vector<short>>& b, unsigned char shft, std::vector<int>& y, int hidden_len) {
    for (int i = 0; i < hidden_len * 4; i++) {
        long long sum = 0;
        for (int j = 0; j < hidden_len; j++) {
            sum += static_cast<long long>(a[j]) * static_cast<long long>(b[j][i]);
        }
        y[i] = srs_int(sum, shft);
    }
}

void scal_int_vec(const std::vector<int>& a, short b, unsigned char shft, std::vector<int>& c, int hidden_len) {
    for (int i = 0; i < hidden_len * 4; i++) {
        long long sum = static_cast<long long>(a[i]) * static_cast<long long>(b);
        c[i] = srs_int(sum, shft);
    }
}

int scal_int(int a, short b, unsigned char shft) {
    long long sum = static_cast<long long>(a) * static_cast<long long>(b);
    return srs_int(sum, shft);
}

short scal_short(int a, unsigned short sc, short zp, unsigned char shft) {
    long long sum = static_cast<long long>(a) * static_cast<long long>(sc) + (static_cast<long long>(zp) << shft);
    return srs_short(sum, shft);
}

void reshape_row_maj(int n_row, int n_col, const unsigned short* xx_i, std::vector<std::vector<short>>& yy_o) {
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            int xx = static_cast<int>(xx_i[i * n_col + j]) - 32768;
            yy_o[i][j] = xx;
        }
    }
}

void reshape_col_maj(int n_row, int n_col, const unsigned short* xx_i, std::vector<std::vector<short>>& yy_o) {
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            int xx = static_cast<int>(xx_i[j * n_row + i]) - 32768;
            yy_o[i][j] = xx;
        }
    }
}

void reshape_vec(int n, const unsigned short* xx_i, std::vector<short>& yy_o) {
    for (int i = 0; i < n; i++) {
        int xx = static_cast<int>(xx_i[i]) - 32768;
        yy_o[i] = xx;
    }
}

int params_float_to_fix(double x_float, short& x_fix) {
    int a = 15 - std::ceil(std::log2(std::fabs(x_float)));
    double s = (1 << a);
    s *= x_float;
    x_fix = round_short(s);
    return a;
}

void get_params(
    const std::vector<std::vector<short>>& w, const std::vector<std::vector<short>>& wb,
    const std::vector<std::vector<short>>& r, const std::vector<std::vector<short>>& rb,
    int input_len, int hidden_len, int blen,
    const std::vector<short>& bx, const std::vector<short>& bh,
    const std::vector<short>& bxb, const std::vector<short>& bhb,
    float x_scale, float w_scale, float r_scale, float b_scale, float y_scale,
    float x_zerop, float w_zerop, float r_zerop, float b_zerop, float y_zerop,
    short& qx, short& qh, short& qa, short& qb, std::vector<int>& qc, std::vector<int>& qcb,
    unsigned short& ohs, unsigned short& ohz,
    unsigned char& xw_shft, unsigned char& xwq_shft, unsigned char& hr_shft, unsigned char& hrq_shft,
    unsigned char& xs_shft, unsigned char& hs_shft, unsigned char& oh_shft
) {
    // Derive parameters
    // Scaling factor for floating-point conversion
    double q_x_float = x_scale * w_scale * s_g;
    double q_h_float = y_scale * r_scale * s_g;
    double q_a_float = q_x_float * w_zerop;
    double q_b_float = q_h_float * r_zerop;

    xw_shft = std::ceil(std::log2(input_len));
    hr_shft = std::ceil(std::log2(hidden_len));

    xwq_shft = params_float_to_fix(q_x_float, qx) - xw_shft;
    hrq_shft = params_float_to_fix(q_h_float, qh) - hr_shft;

    xs_shft = params_float_to_fix(q_a_float, qa);
    hs_shft = params_float_to_fix(q_b_float, qb);

    // qc calculation
    {
        double q_cc = (double)input_len * (double)q_a_float * (double)x_zerop;
        q_cc += (double)hidden_len * (double)q_b_float * (double)y_zerop;

        for (int i = 0; i < blen; i++) {
            double this_qc = q_cc;
            double sum = 0;
            for (int j = 0; j < input_len; j++) sum += (double)w[j][i];
            this_qc -= (double)q_x_float * (double)x_zerop * sum;
            sum = 0;
            for (int j = 0; j < hidden_len; j++) sum += (double)r[j][i];
            this_qc -= (double)q_h_float * (double)y_zerop * sum;
            this_qc += (double)b_scale * s_g * ((double)bx[i] + (double)bh[i] - 2.0 * b_zerop);
            qc[i] = round_int(this_qc);
        }

        for (int i = 0; i < blen; i++) {
            double this_qc = q_cc;
            double sum = 0;
            for (int j = 0; j < input_len; j++) sum += (double)wb[j][i];
            this_qc -= (double)q_x_float * (double)x_zerop * sum;
            sum = 0;
            for (int j = 0; j < hidden_len; j++) sum += (double)rb[j][i];
            this_qc -= (double)q_h_float * (double)y_zerop * sum;
            this_qc += (double)b_scale * s_g * ((double)bxb[i] + (double)bhb[i] - 2.0 * b_zerop);
            qcb[i] = round_int(this_qc);
        }
    }

    {
        double qo = 1.0 / y_scale;
        int a = 16 - std::ceil(std::log2(qo));
        double s = (a >= 0) ? (1 << a) : (1.0 / ((double)(1 << (-a))));
        s *= qo;
        ohs = round_short(s);
        ohz = static_cast<unsigned short>(y_zerop + 0.000000001);
        oh_shft = FLOAT_DEC_POINT + a;
    }
}

void lstm_core(
    std::vector<std::vector<short>>& y, const std::vector<std::vector<short>>& x,
    const std::vector<std::vector<short>>& w, const std::vector<std::vector<short>>& r,
    int seq_len, int input_len, int hidden_len,
    short qx, short qh, short qa, short qb, const std::vector<int>& qc,
    unsigned short ohs, short ohz,
    unsigned char xw_shft, unsigned char xwq_shft, unsigned char hr_shft, unsigned char hrq_shft,
    unsigned char xs_shft, unsigned char hs_shft, unsigned char oh_shft
) {
    std::vector<short> h(hidden_len, ohz);
    std::vector<float> c(hidden_len, 0);

    for (int ite = 0; ite < seq_len; ite++) {
        std::vector<int> xa(4 * hidden_len);
        std::vector<int> xb(4 * hidden_len);
        std::vector<int> xc(4 * hidden_len);

        // xw and hr
        vmul_x(ite, x, w, xw_shft, xa, input_len, hidden_len);
        scal_int_vec(xa, qx, xwq_shft, xb, hidden_len);

        vmul_h(h, r, hr_shft, xa, hidden_len);
        scal_int_vec(xa, qh, hrq_shft, xc, hidden_len);

        // sum x and h
        int sum = 0;
        for (int i = 0; i < input_len; i++) sum += x[ite][i];
        int xs = scal_int(sum, qa, xs_shft);

        sum = 0;
        for (int i = 0; i < hidden_len; i++) sum += h[i];
        int hs = scal_int(sum, qb, hs_shft);

        // add up
        for (int i = 0; i < 4 * hidden_len; i++) xa[i] = xb[i] + xc[i] - xs - hs + qc[i];

        // non-linear functions
        for (int i = 0; i < hidden_len; i++) {
            float xi = xa[i] / (2.0 * s_g);
            float xo = xa[i + hidden_len] / (2.0 * s_g);
            float xf = xa[i + hidden_len * 2] / (2.0 * s_g);
            float xc = xa[i + hidden_len * 3] / s_g;

            float vi = std::tanh(xi) * 0.5 + 0.5;
            float vo = std::tanh(xo) * 0.5 + 0.5;
            float vf = std::tanh(xf) * 0.5 + 0.5;
            float vc = std::tanh(xc);

            float c_float = vi * vc + vf * c[i];
            float h_float = vo * std::tanh(c_float);

            int h_int = round_int(h_float * s_g);

            h[i] = scal_short(h_int, ohs, ohz, oh_shft);
            c[i] = c_float;

            y[ite][i] = h[i];
        }
    }
}

void lstm(
    unsigned short* y_o, const unsigned short* x_i,
    unsigned short* w_i, const unsigned short* r_i, 
    const unsigned short* b_i,
    float x_scale, float w_scale, float r_scale, float b_scale, float y_scale,
    unsigned short x_zerop_i, unsigned short w_zerop_i, unsigned short r_zerop_i, unsigned short b_zerop_i, unsigned short y_zerop_i,
    int seq_len, int input_len, int hidden_len
) {
    int ylen = seq_len * hidden_len;
    int xlen = seq_len * input_len;
    int wlen = input_len * hidden_len * 4;
    int rlen = hidden_len * hidden_len * 4;
    int blen = hidden_len * 4;
    
    // Reshape and conversion
    std::vector<std::vector<short>> x(seq_len, std::vector<short>(input_len));
    std::vector<std::vector<short>> w(input_len, std::vector<short>(hidden_len * 4));
    std::vector<std::vector<short>> r(hidden_len, std::vector<short>(hidden_len * 4));
    std::vector<std::vector<short>> xb(seq_len, std::vector<short>(input_len));
    std::vector<std::vector<short>> wb(input_len, std::vector<short>(hidden_len * 4));
    std::vector<std::vector<short>> rb(hidden_len, std::vector<short>(hidden_len * 4));    
    std::vector<short> bx(hidden_len * 4);
    std::vector<short> bh(hidden_len * 4);
    std::vector<short> bxb(hidden_len * 4);
    std::vector<short> bhb(hidden_len * 4);

    reshape_row_maj(seq_len, input_len, x_i, x);
    reshape_col_maj(input_len, blen, w_i, w);
    reshape_col_maj(hidden_len, blen, r_i, r);
    reshape_vec(blen, b_i, bx);
    reshape_vec(blen, b_i + blen, bh);

    reshape_col_maj(input_len, blen, w_i + wlen, wb);
    reshape_col_maj(hidden_len, blen, r_i + rlen, rb);
    reshape_vec(blen, b_i + blen * 2, bxb);
    reshape_vec(blen, b_i + blen * 3, bhb);

    short qx, qh, qa, qb;
    std::vector<int> qc(blen), qcb(blen);
    unsigned short ohs, ohz;
    unsigned char xw_shft, xwq_shft, hr_shft, hrq_shft, xs_shft, hs_shft, oh_shft;

    get_params(w, wb, r, rb, input_len, hidden_len, blen, bx, bh, bxb, bhb,
        x_scale, w_scale, r_scale, b_scale, y_scale,
        x_zerop_i - 32768, w_zerop_i - 32768, r_zerop_i - 32768, b_zerop_i - 32768, y_zerop_i - 32768,
        qx, qh, qa, qb, qc, qcb, ohs, ohz,
        xw_shft, xwq_shft, hr_shft, hrq_shft,
        xs_shft, hs_shft, oh_shft);

    // Forward lstm
    std::vector<std::vector<short>> ypre(seq_len, std::vector<short>(hidden_len));
    lstm_core(ypre, x, w, r, seq_len, input_len, hidden_len, qx, qh, qa, qb, qc, ohs, ohz, xw_shft, xwq_shft, hr_shft, hrq_shft, xs_shft, hs_shft, oh_shft);

    // Copy results out
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < hidden_len; j++) {
            y_o[i * 2 * hidden_len + j] = static_cast<unsigned short>(ypre[i][j] + 32768);
        }
    }

    // Reversed order lstm
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < input_len; j++) {
            xb[i][j] = x[seq_len - 1 - i][j];
        }
    }

    lstm_core(ypre, xb, wb, rb, seq_len, input_len, hidden_len, qx, qh, qa, qb, qcb, ohs, ohz, xw_shft, xwq_shft, hr_shft, hrq_shft, xs_shft, hs_shft, oh_shft);

    // Copy results out
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < hidden_len; j++) {
            y_o[i * 2 * hidden_len + hidden_len + j] = static_cast<unsigned short>(ypre[seq_len - 1 - i][j] + 32768);
        }
    }
}
