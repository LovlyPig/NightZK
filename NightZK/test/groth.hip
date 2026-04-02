#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp"
#include "libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/examples/run_r1cs_gg_ppzksnark.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

struct GrothProof {
    libff::G1<ppT> evaluation_At, evaluation_Ht, evaluation_Lt;
    libsnark::knowledge_commitment<libff::G2<ppT>, libff::G1<ppT> > evaluation_Bt;
};

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    assert(argc > 1);
    size_t log_m = atoi(argv[1]);
    size_t m = 1 << log_m;
    size_t k = min(m, size_t(1024));
    if (argc > 2) k = atoi(argv[2]);
    m -= (k + 1);

    libff::enter_block("Generate R1CS example");
    libsnark::r1cs_example<libff::Fr<ppT>> example = libsnark::generate_r1cs_example_with_field_input<libff::Fr<ppT>>(m, k);
    libff::leave_block("Generate R1CS example");

    libff::print_header("(enter) Profile R1CS GG-ppzkSNARK");

    libff::enter_block("Call to run_r1cs_gg_ppzksnark");

    libff::print_header("R1CS GG-ppzkSNARK Generator");
    libsnark::r1cs_gg_ppzksnark_keypair<ppT> keypair = libsnark::r1cs_gg_ppzksnark_generator<ppT>(example.constraint_system);
    printf("\n"); libff::print_indent(); libff::print_mem("after generator");

    libff::print_header("Preprocess verification key");
    libsnark::r1cs_gg_ppzksnark_processed_verification_key<ppT> pvk = libsnark::r1cs_gg_ppzksnark_verifier_process_vk<ppT>(keypair.vk);

    libff::print_header("R1CS GG-ppzkSNARK Prover");
    libsnark::r1cs_gg_ppzksnark_proof<ppT> proof = libsnark::r1cs_gg_ppzksnark_prover<ppT>(keypair.pk, example.primary_input, example.auxiliary_input);
    printf("\n"); libff::print_indent(); libff::print_mem("after prover");

    libff::print_header("R1CS GG-ppzkSNARK Verifier");
    const bool ans = libsnark::r1cs_gg_ppzksnark_verifier_strong_IC<ppT>(keypair.vk, example.primary_input, proof);
    printf("\n"); libff::print_indent(); libff::print_mem("after verifier");
    printf("* The verification result is: %s\n", (ans ? "PASS" : "FAIL"));

    libff::print_header("R1CS GG-ppzkSNARK Online Verifier");
    const bool ans2 = libsnark::r1cs_gg_ppzksnark_online_verifier_strong_IC<ppT>(pvk, example.primary_input, proof);
    assert(ans == ans2);

    libff::leave_block("Call to run_r1cs_gg_ppzksnark");

    libff::print_header("(leave) Profile R1CS GG-ppzkSNARK");

    return 0;
}