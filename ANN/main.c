/******************************************************************************

====================================================
Network:      Backpropagation Network with Bias Terms and Momentum
====================================================

Application:  Time-Series Forecasting
Prediction of the Annual Number of Sunspots

Author:       Karsten Kutza
Date:         17.4.96

Reference:    D.E. Rumelhart, G.E. Hinton, R.J. Williams
Learning Internal Representations by Error Propagation
in:
D.E. Rumelhart, J.L. McClelland (Eds.)
Parallel Distributed Processing, Volume 1
MIT Press, Cambridge, MA, pp. 318-362, 1986

******************************************************************************/




/******************************************************************************
D E C L A R A T I O N S
******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "train_data.h"
#include "test_data.h"
#include "define.h"

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


typedef struct {                     /* A LAYER OF A NET:                     */
    INT           Units;         /* - number of units in this layer       */
    REAL*         Output;        /* - output of ith unit                  */
    REAL*         Error;         /* - error term of ith unit              */
    REAL**        Weight;        /* - connection weights to ith unit      */
    REAL**        WeightSave;    /* - saved weights for stopped training  */
    REAL**        dWeight;       /* - last weight deltas for momentum     */
} LAYER;

typedef struct {                     /* A NET:                                */
    LAYER**       Layer;         /* - layers of this net                  */
    LAYER*        InputLayer;    /* - input layer                         */
    LAYER*        OutputLayer;   /* - output layer                        */
    REAL          Alpha;         /* - momentum factor                     */
    REAL          Eta;           /* - learning rate                       */
    REAL          Gain;          /* - gain of sigmoid function            */
    REAL          Error;         /* - total net error                     */
} NET;


/******************************************************************************
R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
******************************************************************************/


void InitializeRandoms()
{
    srand(4711);
}


INT RandomEqualINT(INT Low, INT High)
{
    return rand() % (High - Low + 1) + Low;
}


REAL RandomEqualREAL(REAL Low, REAL High)
{
    return ((REAL)rand() / RAND_MAX) * (High - Low) + Low;
}


/******************************************************************************
A P P L I C A T I O N - S P E C I F I C   C O D E
******************************************************************************/


INT Units[NUM_LAYERS] = { N, 10, M };

REAL train_data_[NUM_TRAIN_DATA][N];

REAL test_data_[NUM_TEST_DATA][N];

REAL                  Mean;
REAL                  TrainError;
REAL                  TrainErrorPredictingMean;
REAL                  TestError;
REAL                  TestErrorPredictingMean;

FILE*                 f;


void NormalizeInput()
{
    INT  i;
    INT Min, Max;

    Min = INT_MAX;
    Max = INT_MIN;
    INT * data_ptr = (INT *)train_data;
    for (i = 0; i < NUM_TRAIN_DATA * N; i++) {
        Min = MIN(Min, *data_ptr);
        Max = MAX(Max, *data_ptr);
        data_ptr++;
    }
    Mean = 0;
    data_ptr = (INT *)train_data;
    REAL * data_ptr_ = (REAL *)train_data_;
    for (i = 0; i < NUM_TRAIN_DATA * N; i++) {
        *data_ptr_ = (((REAL)(*data_ptr - Min)) / (Max - Min)) * (HI - LO) + LO;
        Mean += *data_ptr_ / NUM_TRAIN_DATA / N;
        data_ptr++;
        data_ptr_++;
    }

    data_ptr = (INT *)test_data;
    data_ptr_ = (REAL *)test_data_;
    for (i = 0; i < NUM_TEST_DATA * N; i++) {
        *data_ptr_ = (((REAL)(*data_ptr - Min)) / (Max - Min)) * (HI - LO) + LO;
        data_ptr++;
        data_ptr_++;
    }
}

void InitializeApplication(NET* Net)
{
    INT  j, i;
    REAL Out, Err;

    Net->Alpha = 0.5;
    Net->Eta = 0.05;
    Net->Gain = 1;

    NormalizeInput();
    TrainErrorPredictingMean = 0;
    for (j = 0; j <= NUM_TRAIN_DATA; j++) {
        for (i = 0; i < M; i++) {
            Out = target_train_data[j][i];
            Err = Mean - Out;
            TrainErrorPredictingMean += 0.5 * sqr(Err);
        }
    }
    TestErrorPredictingMean = 0;
    for (j = 0; j <= NUM_TEST_DATA; j++) {
        for (i = 0; i < M; i++) {
            Out = target_test_data[j][i];
            Err = Mean - Out;
            TestErrorPredictingMean += 0.5 * sqr(Err);
        }
    }
    f = fopen("BPN.txt", "w");
}


void FinalizeApplication(NET* Net)
{
    fclose(f);
}


/******************************************************************************
I N I T I A L I Z A T I O N
******************************************************************************/


void GenerateNetwork(NET* Net)
{
    INT l, i;

    Net->Layer = (LAYER**)calloc(NUM_LAYERS, sizeof(LAYER*));

    for (l = 0; l < NUM_LAYERS; l++) {
        Net->Layer[l] = (LAYER*)malloc(sizeof(LAYER));

        Net->Layer[l]->Units = Units[l];
        Net->Layer[l]->Output = (REAL*)calloc(Units[l] + 1, sizeof(REAL));
        Net->Layer[l]->Error = (REAL*)calloc(Units[l] + 1, sizeof(REAL));
        Net->Layer[l]->Weight = (REAL**)calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->WeightSave = (REAL**)calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->dWeight = (REAL**)calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->Output[0] = BIAS;

        if (l != 0) {
            for (i = 1; i <= Units[l]; i++) {
                Net->Layer[l]->Weight[i] = (REAL*)calloc(Units[l - 1] + 1, sizeof(REAL));
                Net->Layer[l]->WeightSave[i] = (REAL*)calloc(Units[l - 1] + 1, sizeof(REAL));
                Net->Layer[l]->dWeight[i] = (REAL*)calloc(Units[l - 1] + 1, sizeof(REAL));
            }
        }
    }
    Net->InputLayer = Net->Layer[0];
    Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
    Net->Alpha = 0.9;
    Net->Eta = 0.25;
    Net->Gain = 1;
}


void RandomWeights(NET* Net)
{
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
            }
        }
    }
}


void SetInput(NET* Net, REAL* Input)
{
    INT i;

    for (i = 1; i <= Net->InputLayer->Units; i++) {
        Net->InputLayer->Output[i] = Input[i - 1];
    }
}


void GetOutput(NET* Net, REAL* Output)
{
    INT i;

    for (i = 1; i <= Net->OutputLayer->Units; i++) {
        Output[i - 1] = Net->OutputLayer->Output[i];
    }
}


/******************************************************************************
S U P P O R T   F O R   S T O P P E D   T R A I N I N G
******************************************************************************/


void SaveWeights(NET* Net)
{
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
            }
        }
    }
}


void RestoreWeights(NET* Net)
{
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
            }
        }
    }
}

/******************************************************************************
A C T I V A T I O N   F U N C T I O N
******************************************************************************/
#define SIGMOID             0
#define RELU                1

#define ACTIVATIONFUNCTION  SIGMOID

REAL ComputeActivationFunction(REAL gain, REAL x)
{
#if ACTIVATIONFUNCTION == SIGMOID
    return (1 / (1 + exp(-gain * x)));
#else
    // default: sigmoid
    return (1 / (1 + exp(-gain * x)));
#endif
}

REAL ComputeDerivative(REAL gain, REAL x)
{
#if ACTIVATIONFUNCTION == SIGMOID
    return (gain * x * (1 - x));
#else
    // default: sigmoid
    return (gain * x * (1 - x));
#endif
}
/******************************************************************************
P R O P A G A T I N G   S I G N A L S
******************************************************************************/


void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
    INT  i, j;
    REAL Sum;

    for (i = 1; i <= Upper->Units; i++) {
        Sum = 0;
        for (j = 0; j <= Lower->Units; j++) {
            Sum += Upper->Weight[i][j] * Lower->Output[j];
        }
        Upper->Output[i] = ComputeActivationFunction(Net->Gain, Sum);
    }
}


void PropagateNet(NET* Net)
{
    INT l;

    for (l = 0; l < NUM_LAYERS - 1; l++) {
        PropagateLayer(Net, Net->Layer[l], Net->Layer[l + 1]);
    }
}

/******************************************************************************
B A C K P R O P A G A T I N G   E R R O R S
******************************************************************************/


void ComputeOutputError(NET* Net, REAL* Target)
{
    INT  i;
    REAL Out, Err;

    Net->Error = 0;
    for (i = 1; i <= Net->OutputLayer->Units; i++) {
        Out = Net->OutputLayer->Output[i];
        Err = Target[i - 1] - Out;
        Net->OutputLayer->Error[i] = ComputeDerivative(Net->Gain, Out) * Err;
        Net->Error += 0.5 * sqr(Err);
    }
}


void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
    INT  i, j;
    REAL Out, Err;

    for (i = 1; i <= Lower->Units; i++) {
        Out = Lower->Output[i];
        Err = 0;
        for (j = 1; j <= Upper->Units; j++) {
            Err += Upper->Weight[j][i] * Upper->Error[j];
        }
        Lower->Error[i] = ComputeDerivative(Net->Gain, Out) * Err;
    }
}


void BackpropagateNet(NET* Net)
{
    INT l;

    for (l = NUM_LAYERS - 1; l > 1; l--) {
        BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l - 1]);
    }
}


void AdjustWeights(NET* Net)
{
    INT  l, i, j;
    REAL Out, Err, dWeight;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Out = Net->Layer[l - 1]->Output[j];
                Err = Net->Layer[l]->Error[i];
                dWeight = Net->Layer[l]->dWeight[i][j];
                Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
                Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
            }
        }
    }
}


/******************************************************************************
S I M U L A T I N G   T H E   N E T
******************************************************************************/


void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training)
{
    SetInput(Net, Input);
    PropagateNet(Net);
    GetOutput(Net, Output);

    ComputeOutputError(Net, Target);
    if (Training) {
        BackpropagateNet(Net);
        AdjustWeights(Net);
    }
}


void TrainNet(NET* Net, INT Epochs)
{
    INT  i, n;
    REAL Output[M];

    for (n = 0; n < Epochs*NUM_TRAIN_DATA; n++) {
        i = RandomEqualINT(0, NUM_TRAIN_DATA);
        SimulateNet(Net, &(train_data_[i][0]), Output, &(target_train_data[i][0]), TRUE);
    }
}


void TestNet(NET* Net)
{
    INT  i;
    REAL Output[M];

    TrainError = 0;
    INT count_pass_train = 0;
    fprintf(f, "\nTest net\n");
    for (i = 0; i <= NUM_TRAIN_DATA; i++) {
        SimulateNet(Net, &(train_data_[i][0]), Output, &(target_train_data[i][0]), FALSE);
        TrainError += Net->Error;
        if((Net->OutputLayer->Output[1] >= 0.5 && target_train_data[i][0] == 0.8) || (Net->OutputLayer->Output[1] < 0.5 && target_train_data[i][0] == 0.2))
        {
            //fprintf(f, "\t%d", i);
            count_pass_train++;
        }
    }
    fprintf(f, "\n");
    TestError = 0;
    INT count_pass_test = 0;
    for (i = 0; i <= NUM_TEST_DATA; i++) {
        SimulateNet(Net, &(test_data_[i][0]), Output, &(target_test_data[i][0]), FALSE);
        TestError += Net->Error;
        if ((Net->OutputLayer->Output[1] >= 0.5 && target_test_data[i][0] == 0.8) || (Net->OutputLayer->Output[1] < 0.5 && target_test_data[i][0] == 0.2))
        {
            //fprintf(f, "\t%d", i);
            count_pass_test++;
        }
    }
    fprintf(f, "\nNMSE is %0.3f, %d on Training Set and %0.3f, %d on Test Set",
        TrainError, count_pass_train,
        TestError, count_pass_test);
}

void DumpNet(NET* Net)
{
    INT l, i, j;
    FILE * file;
    file = fopen("dump_net.txt", "w");
    fprintf(file, "Number of layers: %d\n", NUM_LAYERS);
    fprintf(file, "Input layer: %d node\n", Units[0]);
    for (i = 0; i < NUM_LAYERS - 2; i++)
    {
        fprintf(file, "Hidden layer %d: %d node\n", i, Units[i + 1]);
    }
    
    fprintf(file, "Output layer: %d node\n", Units[NUM_LAYERS - 1]);

    for (l = 1; l < NUM_LAYERS; l++) {
        fprintf(file, "Layer %d:\n", l);
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            fprintf(file, "{\n");
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                fprintf(file, "%f, \t", Net->Layer[l]->Weight[i][j]);
            }
            fprintf(file, "\n},\n");
        }
    }

    fclose(file);
}

void EvaluateNet(NET* Net)
{
   /* INT  Year;
    REAL Output[M];
    REAL Output_[M];

    fprintf(f, "\n\n\n");
    fprintf(f, "Year    Sunspots    Open-Loop Prediction    Closed-Loop Prediction\n");
    fprintf(f, "\n");
    for (Year = EVAL_LWB; Year <= EVAL_UPB; Year++) {
        SimulateNet(Net, &(Sunspots[Year - N]), Output, &(Sunspots[Year]), FALSE);
        SimulateNet(Net, &(Sunspots_[Year - N]), Output_, &(Sunspots_[Year]), FALSE);
        Sunspots_[Year] = Output_[0];
        fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
            FIRST_YEAR + Year,
            Sunspots[Year],
            Output[0],
            Output_[0]);
    }*/
}


/******************************************************************************
M A I N
******************************************************************************/


void main()
{
    NET  Net;
    BOOL Stop;
    REAL MinTestError;

    InitializeRandoms();
    GenerateNetwork(&Net);
    RandomWeights(&Net);
    InitializeApplication(&Net);

    Stop = FALSE;
    MinTestError = MAX_REAL;
    do {
        TrainNet(&Net, 10);
        TestNet(&Net);
        if (TestError < MinTestError) {
            fprintf(f, " - saving Weights ...");
            MinTestError = TestError;
            SaveWeights(&Net);
        }
        else if (TestError > 2 * MinTestError) {
            fprintf(f, " - stopping Training and restoring Weights ...");
            Stop = TRUE;
            RestoreWeights(&Net);
        }
    } while (NOT Stop);

    DumpNet(&Net);
    TestNet(&Net);
    EvaluateNet(&Net);

    FinalizeApplication(&Net);

    system("pause");
}