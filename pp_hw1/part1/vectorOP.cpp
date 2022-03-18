#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // array mask
    if(i + VECTOR_WIDTH > N)
      maskAll = _pp_init_ones(N % VECTOR_WIDTH);
    else
      maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x, result;
  __pp_vec_float one = _pp_vset_float(1.f);
  __pp_vec_float maxVal = _pp_vset_float(9.999999f);
  __pp_vec_int exp;
  __pp_vec_int zero = _pp_vset_int(0), one_int = _pp_vset_int(1);
  __pp_mask maskAll, maskTmp, maskExpZero, maskRem;
  for(int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // array mask
    if(i + VECTOR_WIDTH > N)
      maskAll = _pp_init_ones(N % VECTOR_WIDTH);
    else
      maskAll = _pp_init_ones();

    // load vector of values
    _pp_vload_int(exp, exponents + i, maskAll);
    // store value for exp == 0
    _pp_veq_int(maskExpZero, exp, zero, maskAll);         // if(exp == 0)
    _pp_vstore_float(output + i, one, maskExpZero);       // output = 1.f
    // mask for else (nonzero & in range)
    maskRem = _pp_mask_not(maskExpZero);
    maskAll = _pp_mask_and(maskAll, maskRem);         
    _pp_vload_float(result, values + i, maskAll);     // result = values
    _pp_vload_float(x, values + i, maskAll);          // x = values
    _pp_vgt_int(maskTmp, exp, one_int, maskAll);      // mask for exponent larger than one
    while(_pp_cntbits(maskTmp) > 0) {                 // while(count > 0)
      _pp_vmult_float(result, result, x, maskTmp);    // check exponent
      _pp_vsub_int(exp, exp, one_int, maskTmp);       // update new exponent
      _pp_vgt_int(maskTmp, exp, one_int, maskTmp);    // find mask for next round
    }
    _pp_vstore_float(output + i, result, maskAll);
    _pp_vgt_float(maskTmp, result, maxVal, maskAll);  // mask for result larger than 9.999f
    _pp_vstore_float(output + i, maxVal, maskTmp);    // clamp value
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float x;
  __pp_vec_float sum = _pp_vset_float(0.f);
  __pp_vec_float zeros = _pp_vset_float(0.f);
  __pp_vec_float ones = _pp_vset_float(1.f);
  __pp_mask maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // load value
    _pp_vload_float(x, values + i, maskAll);
    // add into sum
    _pp_vadd_float(sum, sum, x, maskAll);
  }
  for(int cnt = _pp_cntbits(maskAll); cnt > 1;) {
    _pp_hadd_float(sum, sum);
    _pp_interleave_float(sum, sum);
    cnt >>= 1;
    maskAll = _pp_init_ones(cnt);
  }
  return sum.value[0];
}