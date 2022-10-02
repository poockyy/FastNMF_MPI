#include <mpi.h>
#include <string.h>
#include <vector>

using namespace std;
//#include <FNMF.h> 

//[[Rcpp::depends(RcppEigen)]]
#include<RcppEigen.h>
/*

RUN WITH
mpirun -np 2 ./a.out
mpirun [ -np X ] [ --hostfile <filename> ] <program>
X ∈ the number of processes you want to run of this



MPI_Gather(SENT, # SENT , MPI_DATA, RECV, # RECV, MPI_DATA, ROOT, MPI_COMM_WORLD);

Tasks:
    1) read in chunks of A to each node -> DO THROUGH RCPP::Exports
        - Should take as input a list of files to read into each node
    2) row-wise splice each chunk in its node
    3) transpose **copy** each row
    4) distrbute each column to it's correct row
    5) collect nodes together as a transpose of A
    6.) ????
    6) profit
*/

int main(int argc, char** argv) {

    Eigen::MatrixXd chunk;
  

    // Initialize MPI Environment
    MPI_Init(&argc, &argv);

    // Get # of processes associated with the communicator
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get the rank of the calling process
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    /*------------------------------------------------------------------------------------------------*/

    // Read in chunks of A to each process
    if(worldRank == 0){ //rank of 0 is master thread
        vector<Eigen::MatrixXd>* chunks;
        for(int i = 0; i < worldSize; i++){ //master thread sends all other threads the chunks
            MPI_Ssend(chunks[i], 1, MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }

    } else { //other threads are not worldRank 0, so they receive
        MPI_Recv(&chunk, 1, MPI_BYTE, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    
    /* WORKING EXAMPLE OF WHAT NEEDS TO BE DONE
    char* str1 = "NMF";

    for(int i = 0; i < strlen(str1); i++){
        //printf("%c\n", str1[i]);
        MPI_Send(&str1[i], 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    char* c;
    for(int i = 0; i < strlen(str1); i++){
        MPI_Recv(&c, 1, MPI_CHAR, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%d received %c from %d\n", worldRank, c, i);
    }
    */


    //Gets the transpose and sends columns to their righful node
    if(worldRank != 0){
        Eigen::MatrixXd newChunk = AAt(chunk);
        Eigen::VectorXd tCol; //the received column
        int col = 0;
        while(col < newChunk.cols()){
            //Sends out as many columns as there are ranks. I.e 8 cores committed to this 
            //will send out 8 columns of the transposed matrix at a time
            for(col = 0; col < worldSize; col++){

                 // distribute transpose rows to correct node. Each column goes to a specific world rank
                MPI_Ssend(newChunk.col(col), 1, MPI_BYTE, col, 0, MPI_COMM_WORLD); 
            }

            //That worldRank picks up that column to use in the next step.
            MPI_Recv(&tCol, 1, MPI_BYTE, worldRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      }
    }

    
    //Step 4) distrbute each column to it's correct row

    



    // Profit!

    // collect nodes together, will look something like line below
    // MPI_Gather(chunk, 1, MPI_CHAR, chunks[index], worldSize, MPI_BYTE, 0, MPI_COMM_WORLD);

    /* WORKING EXAMPLE OF A FUNCTIONAL GATHER
    char* send_str1 = "NMF";
    char rcv_word[5];
    
    MPI_Send(&str1, 5, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    
    int* buffer = (int*)malloc(worldSize*sizeof(int));
    MPI_Gather(str1, 5, MPI_CHAR, word, 5, MPI_BYTE, 0, MPI_COMM_WORLD);

    printf("%d received %s\n", worldRank, word);
    free(buffer);
    */

    /*------------------------------------------------------------------------------------------------*/

    // Free MPI Environment
    MPI_Finalize();
    
    
    Eigen::MatrixXd AAt(const Eigen::MatrixXd& A) {
      Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(A.rows(), A.rows());
      AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
      AAt.triangularView<Eigen::Upper>() = AAt.transpose();
      AAt.diagonal().array() += 1e-15;  // for numerical stability during coordinate descent NNLS
      return AAt;
    }

}


