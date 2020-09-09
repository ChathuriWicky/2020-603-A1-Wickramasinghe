//make -f Makefile
//mpirun -np 1 ./main datasets/small.arff

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdint.h>
#include <iterator>
#include<algorithm>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include <iostream>
#include <string>
#include <bitset>
#include <time.h>
#include <map>
#include <vector>
#include <set>
#include<list>
#include<random>

#include <mpi.h>



using namespace std;

int get_mode(int* class_array, int class_array_size) {

    int* ipRepetition = new int[class_array_size];
    for (int i = 0; i < class_array_size; ++i) {
        ipRepetition[i] = 0;
        int j = 0;
        bool bFound = false;
        while ((j < i) && (class_array[i] != class_array[j])) {
            if (class_array[i] != class_array[j]) {
                ++j;
            }
        }
        ++(ipRepetition[j]);
    }
    int index_max_repeated = 0;
    for (int i = 1; i < class_array_size; ++i) {
        if (ipRepetition[i] > ipRepetition[index_max_repeated]) {
            index_max_repeated = i;
        }
    }
    delete [] ipRepetition;
    return class_array[index_max_repeated];

}

int find_class(float* distances_temp, ArffData* dataset,  int K ){
  //cout << "in the function----------------------------" << "\n";

  int no_of_data_records = dataset->num_instances();

  //std::cout << "no_of_data_records: " << no_of_data_records <<" \n";

  vector<float> distances(distances_temp, distances_temp + no_of_data_records);


    vector<int> index(distances.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (distances[a] < distances[b]);
        }
    );

    int* predictions = (int*)malloc(K * sizeof(int));

    for (int i = 0 ; i< K ; i++) {
        //cout << index[i] << endl;
        predictions[i] = dataset->get_instance(index[i])->get(dataset->num_attributes() - 1)->operator int32();
        //std::cout << " predictions[i]  " << predictions[i]  << "\n";
    }

    int predicted_class = get_mode(predictions, K) ;
    //std::cout << "smallestDistance in the function" <<  distances_temp[index[0]]<< "\n Function end \n";
    //std::cout << "predicted_class" << predicted_class << "\n";
    return predicted_class;
}

int* KNN(ArffData* dataset, int K, int start_pos, int stop_pos)
{

  cout << "K :" << K << "\n";
  int no_of_datapoints = (stop_pos - start_pos) + 1;
  int* predictions = (int*)malloc(no_of_datapoints * sizeof(int));


  int temp=0;
  for(int i = start_pos; i < stop_pos+1; i++) // for each instance in the dataset
  {

      float smallestDistance = FLT_MAX;
      int smallestDistanceClass;
      float* distances = (float*)malloc(dataset->num_instances() * sizeof(float));
      distances[start_pos+ temp] = FLT_MAX;


      for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
      {
          if( start_pos+ temp == j) continue;

          float distance = 0;

          for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
          {
              float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
              distance += diff * diff;
          }

          distance = sqrt(distance);
          distances[j] = distance;
          if(distance < smallestDistance) // select the closest one
          {
              smallestDistance = distance;
              smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
          }
      }




      int temp_class = find_class(distances, dataset, K);
      //std::cout << "closest class from the function" << dataset->get_instance(temp_class)->get(dataset->num_attributes() - 1)->operator int32() << "\n";
      //std::cout << "smallestDistanceClass from KNN" << smallestDistanceClass << '\n';
      predictions[temp] = temp_class; //smallestDistanceClass;
      temp++;
      //std::cout << "For data point: " << i << " smallestDistanceClass: " << smallestDistanceClass << " Predicted with given K : " <<  temp_class << "\n";

  }

  return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

void print_elements(int* array, int size){
  std::cout << "Printing local elements" << "\n";
  for(int i=0;i<size;i++){
    std::cout << array[i] << " , ";
  }

}

void fill_main_predictions(int* predictions_main, int* predictions_local, int start_pos, int stop_pos){
  int temp=0;
  for(int i= start_pos; i<=stop_pos; i++){
    predictions_main[i] = predictions_local[temp];
    temp++;
  }

}

int main(int argc, char *argv[])
{

    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }

    int size;
    int rank;
    MPI_Status status;
    const int VERY_LARGE_INT = 999999;
    const int ROOT = 0;
    int tag = 1234;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status Stat;





    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    int K = 1;


    if (rank == ROOT) {

      int no_of_datapoints = dataset->num_instances();
      int count = no_of_datapoints/(size);
      int remainder = no_of_datapoints % size;
      std::cout << "Im root with rank " << rank << "  ::: " << "count per p is " << count << " No of datapoints:" << no_of_datapoints << "\n";


        if( size ==1) {

          int start_pos = 0;
          int stop_pos = count - 1;

          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          // Get the class predictions
          int* predictions = KNN(dataset, K, start_pos, stop_pos);
          //print_elements(predictions, no_of_datapoints);
          // Compute the confusion matrix
          int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
          // Calculate the accuracy
          float accuracy = computeAccuracy(confusionMatrix, dataset);

          clock_gettime(CLOCK_MONOTONIC_RAW, &end);
          uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

          printf("The KNN classifier, Serial Version for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);

        }
        else{
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);

            for(int i = 1; i < size; i++){
                int start_pos, stop_pos;
                start_pos = i * (count);
                if ( i==size-1) {
                    stop_pos = start_pos + count- 1 + remainder;
                } else {
                    stop_pos = start_pos + (count - 1);
                }
                std::cout << "start_pos: " << start_pos << " end:" << stop_pos << "\n";

                MPI_Send(&start_pos, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&stop_pos, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
            int start_pos = 0;
            int stop_pos = count - 1;
            std::cout << "Im the root start_pos: " << start_pos << " end:" << stop_pos << "\n";

            int* predictions_main = (int*)malloc(no_of_datapoints * sizeof(int));


            // for root, calc local predictions
            int datapoints_count = (stop_pos - start_pos) + 1;
            int* local_predictions = KNN(dataset, K, start_pos, stop_pos);
            //print_elements(local_predictions, datapoints_count);
            fill_main_predictions(predictions_main, local_predictions, start_pos, stop_pos);


            for(int i = 1; i < size; i++){
                int start_pos, stop_pos;
                start_pos = i * (count);
                if ( i==size-1) {
                    stop_pos = start_pos + count- 1 + remainder;
                } else {
                    stop_pos = start_pos + (count - 1);
                }
                int no_of_ele = (stop_pos - start_pos) + 1;
                int *localArray = (int *)malloc(no_of_ele * sizeof(int));
                MPI_Recv(localArray, no_of_ele, MPI_INT, i, 3, MPI_COMM_WORLD, &Stat);
                fill_main_predictions(predictions_main, localArray, start_pos, stop_pos);
                //print_elements(predictions_main, no_of_ele);

            }

            //print_elements(predictions_main, no_of_datapoints);

            int* confusionMatrix = computeConfusionMatrix(predictions_main, dataset);
            // Calculate the accuracy
            float accuracy = computeAccuracy(confusionMatrix, dataset);

            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

            printf("The KNN classifier, Parallel Version for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);



        }





    } // end of root

    else{

        int start_pos, stop_pos;
        MPI_Recv(&start_pos, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &Stat);
        MPI_Recv(&stop_pos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &Stat);
        int no_of_ele = (stop_pos - start_pos) + 1;

        std::cout << "Im rank " << rank << " with start_pos: " << start_pos << " end:" << stop_pos << " No of ele to process: " << no_of_ele << "\n";

        int* local_predictions = KNN(dataset, K, start_pos, stop_pos);
        //print_elements(local_predictions, no_of_ele);
        MPI_Send(local_predictions, no_of_ele, MPI_INT, 0, 3, MPI_COMM_WORLD);


    } // end of other processors

    MPI_Finalize();
    return 0;

}
