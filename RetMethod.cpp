/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */



#include "RetMethod.h"
#include "Param.hpp"
#include "common_headers.hpp"
#include <cmath>
#include "DocUnigramCounter.hpp"
#include "RelDocUnigramCounter.hpp"
#include "IndexManager.hpp"
#include "OneStepMarkovChain.hpp"
#include <vector>
#include <set>
#include <algorithm>
#include <FreqVector.hpp>

#include <fstream>
#include <sstream>



using namespace lemur::api;
using namespace lemur::retrieval;
using namespace lemur::utility;

bool pairCompare(const std::pair<double, int>& firstElem, const std::pair<double, int>& secondElem);
struct sort_pred{
    bool operator()(const std::pair<int,float> &left,const std::pair<int,float> &right){
        return left.second > right.second;
    }
};
extern int RSMethodHM; // 0--> LM , 1--> RecSys
extern int negGenModeHM;//0 --> coll , 1--> nonRel
extern int feedbackMode;
extern int updatingThresholdMode; // 0 ->no updating ,1->linear

extern map<int,vector<double> >wordEmbedding;
extern map<int,vector<double> >docIdKeyWords;
extern set<int> stopWords;
//extern map<int, map<int,double> > FEEDBACKMAP;
extern vector<pair<int ,vector<double> > > queryTermsIdVec;
extern vector<pair<int, double> >weightedQueryTerms;


static int qid=1;
static string RM;
Index* myIndex = NULL;
void lemur::retrieval::QueryModel::interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                                   double origModCoeff,
                                                   int howManyWord,
                                                   double prSumThresh,
                                                   double prThresh) {
    if (!qm) {
        qm = new lemur::api::IndexedRealVector();
    } else {
        qm->clear();
    }

    qModel.startIteration();
    while (qModel.hasMore()) {
        IndexedReal entry;
        qModel.nextWordProb((TERMID_T &)entry.ind,entry.val);
        qm->push_back(entry);

    }
    qm->Sort();

    double countSum = totalCount();

    // discounting the original model
    startIteration();
    while (hasMore()) {
        QueryTerm *qt = nextTerm();
        incCount(qt->id(), qt->weight()*origModCoeff/countSum);
        delete qt;
    }

    // now adding the new model
    double prSum = 0;
    int wdCount = 0;
    IndexedRealVector::iterator it;
    it = qm->begin();
    while (it != qm->end() && prSum < prSumThresh &&
           wdCount < howManyWord && (*it).val >=prThresh) {
        incCount((*it).ind, (*it).val*(1-origModCoeff));
        prSum += (*it).val;
        it++;
        wdCount++;
    }

    //Sum w in Q qtf * log(qtcf/termcount);
    colQLikelihood = 0;
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::load(istream &is)
{
    // clear existing counts
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        setCount(qt->id(),0);
    }
    colQLikelihood = 0;

    int count;
    is >> count;
    char wd[500];
    double pr;
    while (count-- >0) {
        is >> wd >> pr;
        TERMID_T id = ind.term(wd);
        if (id != 0) setCount(id, pr); // don't load OOV terms
    }
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::save(ostream &os)
{
    int count = 0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        delete qt;
    }
    os << " " << count << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        os << ind.term(qt->id()) << " "<< qt->weight() << endl;
        delete qt;
    }
}

void lemur::retrieval::QueryModel::clarity(ostream &os)
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // clarity should be computed with log_2, so divide by log(2).
    os << "=" << count << " " << (ln_Pr/(sum ? sum : 1.0)/log(2.0)) << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        // print clarity for each query term
        // clarity should be computed with log_2, so divide by log(2).
        os << ind.term(qt->id()) << " "
           << (qt->weight()*log(qt->weight()/
                                ((double)ind.termCount(qt->id())/
                                 (double)ind.termCount())))/log(2.0) << endl;
        delete qt;
    }
}

double lemur::retrieval::QueryModel::clarity() const
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // normalize by sum of probabilities in the input model
    ln_Pr = ln_Pr/(sum ? sum : 1.0);
    // clarity should be computed with log_2, so divide by log(2).
    return (ln_Pr/log(2.0));
}

lemur::retrieval::RetMethod::RetMethod(const Index &dbIndex,
                                       const string &supportFileName,
                                       ScoreAccumulator &accumulator) :
    TextQueryRetMethod(dbIndex, accumulator), supportFile(supportFileName) {

    //docParam.smthMethod = RetParameter::defaultSmoothMethod;
    docParam.smthMethod = RetParameter::DIRICHLETPRIOR;
    //docParam.smthMethod = RetParameter::ABSOLUTEDISCOUNT;
    //docParam.smthMethod = RetParameter::JELINEKMERCER;

    docParam.smthStrategy= RetParameter::defaultSmoothStrategy;
    //docParam.ADDelta = RetParameter::defaultADDelta;
    docParam.JMLambda = RetParameter::defaultJMLambda;
    //docParam.JMLambda = 0.9;
    docParam.DirPrior = 1000;//dbIndex.docLengthAvg();//50;//RetParameter::defaultDirPrior;

    qryParam.adjScoreMethod = RetParameter::NEGATIVEKLD;
    //qryParam.adjScoreMethod = RetParameter::QUERYLIKELIHOOD;
    //qryParam.fbMethod = RetParameter::defaultFBMethod;
    //qryParam.fbMethod = RetParameter::DIVMIN;




    qryParam.fbMethod = RetParameter::RM1;
    RM="RM1";// *** Query Likelihood adjusted score method *** //
    qryParam.fbCoeff = RetParameter::defaultFBCoeff;//default = 0.5
    //qryParam.fbCoeff =0.1;
    qryParam.fbPrTh = RetParameter::defaultFBPrTh;
    qryParam.fbPrSumTh = RetParameter::defaultFBPrSumTh;
    qryParam.fbTermCount = 45;//RetParameter::defaultFBTermCount;//default = 50
    qryParam.fbMixtureNoise = RetParameter::defaultFBMixNoise;
    qryParam.emIterations = 50;//RetParameter::defaultEMIterations;

    docProbMass = NULL;
    uniqueTermCount = NULL;
    mcNorm = NULL;

    NegMu = ind.docLengthAvg();
    collectLMCounter = new lemur::langmod::DocUnigramCounter(ind);
    collectLM = new lemur::langmod::MLUnigramLM(*collectLMCounter, ind.termLexiconID());

    delta = 0.007;

    newNonRelRecieved = false;
    newRelRecieved = false;
    newNonRelRecievedCnt = 0,newRelRecievedCnt =0;


    relMu = 0 ;
    nonRelMu = 0;
    relX2 = 0;
    nonRelX2 = 0 ;


    W2VecDimSize = 100;
    Vq.assign(W2VecDimSize , 0.0);

    scFunc = new ScoreFunc();
    scFunc->setScoreMethod(qryParam.adjScoreMethod);
}
void lemur::retrieval::RetMethod::setCoeffParam(double coe)
{
    qryParam.fbCoeff = coe;
}
lemur::retrieval::RetMethod::~RetMethod()
{
    //delete [] prev_distQuery;
    delete [] docProbMass;
    delete [] uniqueTermCount;
    delete [] mcNorm;
    delete collectLM;
    delete collectLMCounter;
    delete scFunc;

    //delete [] relComputed;//FIX ME!!!!!!!
    //delete [] nonRelComputed;//FIX ME!!!!!!!

    //delete [] coefMatrix[]; //FIX ME!!!!
}

void lemur::retrieval::RetMethod::loadSupportFile() {
    ifstream ifs;
    int i;

    // Only need to load this file if smooth strategy is back off
    // or the smooth method is absolute discount. Don't reload if
    // docProbMass is not NULL.

    if (docProbMass == NULL &&
            (docParam.smthMethod == RetParameter::ABSOLUTEDISCOUNT ||
             docParam.smthStrategy == RetParameter::BACKOFF)) {
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading "
             << supportFile << endl;

        ifs.open(supportFile.c_str());
        if (ifs.fail()) {
            throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                             "smoothing support file open failure");
        }
        COUNT_T numDocs = ind.docCount();
        docProbMass = new double[numDocs+1];
        uniqueTermCount = new COUNT_T[numDocs+1];
        for (i = 1; i <= numDocs; i++) {
            DOCID_T id;
            int uniqCount;
            double prMass;
            ifs >> id >> uniqCount >> prMass;
            if (id != i) {
                throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                 "alignment error in smooth support file, wrong id:");
            }
            docProbMass[i] = prMass;
            uniqueTermCount[i] = uniqCount;
        }
        ifs.close();
    }

    // only need to load this file if the feedback method is
    // markov chain. Don't reload if called a second time.

    if (mcNorm == NULL && qryParam.fbMethod == RetParameter::MARKOVCHAIN) {
        string mcSuppFN = supportFile + ".mc";
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading " << mcSuppFN << endl;

        ifs.open(mcSuppFN.c_str());
        if (ifs.fail()) {
            throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                            "Markov chain support file can't be opened");
        }

        mcNorm = new double[ind.termCountUnique()+1];

        for (i = 1; i <= ind.termCountUnique(); i++) {
            TERMID_T id;
            double norm;
            ifs >> id >> norm;
            if (id != i) {
                throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                "alignment error in Markov chain support file, wrong id:");
            }
            mcNorm[i] = norm;
        }
    }
}

DocumentRep *lemur::retrieval::RetMethod::computeDocRep(DOCID_T docID)
{
    switch (docParam.smthMethod) {
    case RetParameter::JELINEKMERCER:
        return( new JMDocModel(docID,
                               ind.docLength(docID),
                               *collectLM,
                               docProbMass,
                               docParam.JMLambda,
                               docParam.smthStrategy));
    case RetParameter::DIRICHLETPRIOR:
        return (new DPriorDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior,
                                   docParam.smthStrategy));
    case RetParameter::ABSOLUTEDISCOUNT:
        return (new ABSDiscountDocModel(docID,
                                        ind.docLength(docID),
                                        *collectLM,
                                        docProbMass,
                                        uniqueTermCount,
                                        docParam.ADDelta,
                                        docParam.smthStrategy));
    case RetParameter::TWOSTAGE:
        return (new TStageDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior, // 1st stage mu
                                   docParam.JMLambda, // 2nd stage lambda
                                   docParam.smthStrategy));


    default:
        // this should throw, not exit.
        cerr << "Unknown document language model smoothing method\n";
        exit(1);
    }
}


void lemur::retrieval::RetMethod::nearestTerm2Vec(vector<double> vec ,vector<int> relJudDocs,
                                                  vector<pair<int,double> > &nearestTerm)
{

#if 1
    //nearest from relJudgDoc
    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relJudDocs, ind);

    //nearest from coll
    //khob yebar hesab kon!!!!!!
    //lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(ind);

    vector<pair<double,int> >simTermid;
    dCounter->startIteration();
    while(dCounter->hasMore())
    {
        int id;
        double wd;
        dCounter->nextCount(id, wd);

        //if(ind.termCount() < id)
        //    cerr<<"***********************"<<id<<endl;

        vector<double>dtemp;
        if( wordEmbedding.find(id) != wordEmbedding.end() )
            dtemp.assign(wordEmbedding[id].begin() ,wordEmbedding[id].end());
        else
        {
            //cerr<<"no nearest found!";
            continue;
        }

        double sim = this->cosineSim(vec,dtemp);
        //double sim = this->softMaxFunc(vec,dtemp);
        simTermid.push_back(pair<double,int>(sim,id));

    }

    std::sort(simTermid.begin() , simTermid.end(),pairCompare);

    int termCount = min((int)tops4EachQueryTerm , (int)simTermid.size());
    for(int i = 0 ; i < termCount ; i++)//n
    {
        nearestTerm.push_back(make_pair<int ,double > (simTermid[i].second,simTermid[i].first));
        //cerr<<simTermid[i].second<<" ";
    }

    delete dCounter;
#endif

#if 0 //nearest from Coll
    vector<pair<double,int> >simTermid;
    //vector<vector<double> > outVec;
    for(int i =1 ; i< ind.termCountUnique() ; i++)
    {
        vector<double>dtemp;
        if( wordEmbedding.find(i) != wordEmbedding.end() )
            dtemp.assign(wordEmbedding[i].begin() ,wordEmbedding[i].end());
        else
        {
            //cerr<<"no nearest found!";
            continue;
        }

        double sim = this->cosineSim(vec,dtemp);
        //double sim = this->softMaxFunc(vec,dtemp);
        simTermid.push_back(pair<double,int>(sim,i));
    }
    std::sort(simTermid.begin() , simTermid.end(),pairCompare);

    for(int i = 0 ; i < tops4EachQueryTerm ; i++)//n
    {
        nearestTerm.push_back(make_pair<int ,double > (simTermid[i].second,simTermid[i].first));
    }
#endif

}

void lemur::retrieval::RetMethod::computeClipRM1(vector<int> relJudgDoc ,lemur::api::TextQueryRep &origRep )
{
#define RM1SUM 1
#define RM1MNZ 0
#define RM1MAX 0

    //Begin compute RM1
    double collSize = ind.termCount();
    COUNT_T numTerms = ind.termCountUnique();
    double *distQuery = new double[numTerms+1];
    TERMID_T i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;

    for(int i = 0 ; i < relJudgDoc.size() ;i++)
    {
        lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc[i], ind);

        double py = 1.0 ;
        double docLen = dCounter->sum();
        for(int k = 0 ; k < queryTermsIdVec.size() ;k++)
        {
            int qid = queryTermsIdVec[k].first;
            double pdir = (dCounter->count(qid) + (docParam.DirPrior*(ind.termCount(qid)/collSize) ) ) /  (docLen+ docParam.DirPrior);
            //            cerr<<py<<" ";
            py *= pdir ;
        }
        //cerr<<py<<" ";

        //double pSum=0;
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; // dmf FIXME
            double wdPr;

            dCounter->nextCount(wd, wdPr);
            distQuery[wd] += wdPr * py;
            //pSum += wdPr;
        }//end-for-queryRelJudg

        delete dCounter;
    }//end-for-relJudg


    vector<pair<double, int> >scIDVec;
    for(int i = 1 ; i <= numTerms ; i++)
        scIDVec.push_back(make_pair<double,int>(distQuery[i], i ) );

    std::sort(scIDVec.begin() , scIDVec.end() , pairCompare);// can use top n selecting algorithm O(n)

    lemur::utility::ArrayCounter<double> rm1LmCounter(numTerms+1);
    double topSum = 0;
    for (int i = 0; i < topsCinRM1; i++)
        topSum += scIDVec[i].first;

    for (int i = 0; i < topsCinRM1; i++)
        rm1LmCounter.incCount(scIDVec[i].second , scIDVec[i].first/topSum);

    //delete fblm;
    delete distQuery;
    //END RM1


#if RM1SUM
    /// compute p(.|M)

    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;

    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec ,relJudgDoc, nearestIdVecs);//n(50,100) most similar terms


        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );

            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
        }

        int hmin = temp.size();
        if(hmin > tops4EachQueryTerm )
            hmin = tops4EachQueryTerm;

        for(int j = 0 ; j < hmin ; j++)//////
        {
            temp[j].first /= totalSc;

            /*
            double idf1 = log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first ) );
            double idf2= log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first) ) /(log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first) )+1 ) ;
            double dist = weightedQueryTerms[i].second;
            double expdist = exp(dist);
            */

            idScore[temp[j].second] += temp[j].first  ;
        }
    }

    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
        probWordVec.push_back(make_pair<double,int>(mit->second, mit->first) );

    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)

    lemur::utility::ArrayCounter<double> sumLmCounter(numTerms+1);

    double hmin = probWordVec.size() ;
    if(hmin > topsCinRM1)
        hmin = topsCinRM1;

    for (int i = 0; i < hmin; i++)//c//v//(50,100)
        sumLmCounter.incCount(probWordVec[i].second , probWordVec[i].first);

    //end
#endif

#if RM1MNZ
    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;
    map<int, int>tidNumOfOccurInQueryNearestList;


    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec , nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            tidNumOfOccurInQueryNearestList[nearestIdVecs[j].first ]++;
        }

        for(int j = 0 ; j < tops4EachQueryTerm ; j++)
        {
            temp[j].first /= totalSc;

            idScore[temp[j].second] += temp[j].first;

        }



        //for(int ii = i * tops4EachQueryTerm ; ii < probWordVec.size() ; ii++)
        //probWordVec[ii].first /= totalSc;
        //cerr<<ind.term(queryTermsIdVec[i].first)<<" ";
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
    {
        double hscore = mit->second * (tidNumOfOccurInQueryNearestList[mit->first]);
        probWordVec.push_back(make_pair<double,int>(hscore, mit->first) );
    }


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)
    /*cerr<<" ::: ";
    for(int i = 0 ; i < probWordVec.size() ;i ++)
        cerr<< ind.term(probWordVec[i].second)<<" ";
    cerr << endl;*/

    //COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> sumLmCounter(numTerms+1);

    for (int i = 0; i < tops4EachQuery; i++)//v
        sumLmCounter.incCount(probWordVec[i].second , probWordVec[i].first);

#endif

#if RM1MAX
    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;//idMaxScore


    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec , nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
        }

        for(int j = 0 ; j < tops4EachQueryTerm ; j++)
        {
            temp[j].first /= totalSc;

            if( idScore[temp[j].second] < temp[j].first )
                idScore[temp[j].second] = temp[j].first;
        }


        //for(int ii = i * tops4EachQueryTerm ; ii < probWordVec.size() ; ii++)
        //probWordVec[ii].first /= totalSc;
        //cerr<<ind.term(queryTermsIdVec[i].first)<<" ";
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
    {
        probWordVec.push_back(make_pair<double,int>(mit->second, mit->first) );
    }


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)
    /*cerr<<" ::: ";
    for(int i = 0 ; i < probWordVec.size() ;i ++)
        cerr<< ind.term(probWordVec[i].second)<<" ";
    cerr << endl;*/

    //COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> sumLmCounter(numTerms+1);

    for (int i = 0; i < tops4EachQuery; i++)//v
        sumLmCounter.incCount(probWordVec[i].second , probWordVec[i].first);

#endif


    //begin mix
    //cerr<<"PROB:\n";

    lemur::utility::ArrayCounter<double> mixLmCounter(numTerms+1);
    for(int i = 1 ; i<= numTerms ; i++)
    {

        double prob = sumLmCounter.count(i) *alphaCoef +(1-alphaCoef)* rm1LmCounter.count(i);
        //cerr<<prob<<" ";
        mixLmCounter.incCount(i,prob);

        //cerr<<prob<<" ";
    }
    //cerr<<endl;


    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *mixFblm = new lemur::langmod::MLUnigramLM(mixLmCounter, ind.termLexiconID());
    qr->interpolateWith(*mixFblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    delete mixFblm;

    //end mix


}
void lemur::retrieval::RetMethod::updateProfile(lemur::api::TextQueryRep &origRep,
                                                vector<int> relJudgDoc ,vector<int> nonRelJudgDoc)
{
#define BL 0
    /**mine methods****/
#define LOGLOGISTIC 1
#define COSREL 0
    /**other methods***/
#define CENTROID 0
#define COMBSUM 0
#define COMBMNZ 0
#define COMBMAX 0

#define RMONE 0

    /*if(relJudgDoc.size()== 0 )//if we use relJudDoc for nearest term selection!!!!!
    {
        cerr<<"RELJUDGGGGGGGGG!\n";
        return;
    }*/

#if BL
    IndexedRealVector rel;
    for (int i =0 ; i<relJudgDoc.size() ; i++)
    {
        rel.PushValue(relJudgDoc[i],0);
    }
    PseudoFBDocs  *relDocs;
    relDocs= new PseudoFBDocs(rel,relJudgDoc.size(),true);

    updateTextQuery(origRep, *relDocs ,*relDocs);
#endif

#if RMONE

    computeClipRM1(relJudgDoc ,origRep);


#endif

#if COMBMAX
    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;//idMaxScore


    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec ,relJudgDoc, nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
        }

        for(int j = 0 ; j < nearestIdVecs.size()/*tops4EachQueryTerm*/ ; j++)
        {
            temp[j].first /= totalSc;

            if( idScore[temp[j].second] < temp[j].first )
                idScore[temp[j].second] = temp[j].first;
        }


        //for(int ii = i * tops4EachQueryTerm ; ii < probWordVec.size() ; ii++)
        //probWordVec[ii].first /= totalSc;
        //cerr<<ind.term(queryTermsIdVec[i].first)<<" ";
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
    {
        probWordVec.push_back(make_pair<double,int>(mit->second, mit->first) );
    }


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)
    /*cerr<<" ::: ";
    for(int i = 0 ; i < probWordVec.size() ;i ++)
        cerr<< ind.term(probWordVec[i].second)<<" ";
    cerr << endl;*/

    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    for (int i = 0; i < tops4EachQuery; i++)//v
        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);


    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    delete fblm;


#endif
#if COMBMNZ

    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;
    map<int, int>tidNumOfOccurInQueryNearestList;


    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec ,relJudgDoc, nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            tidNumOfOccurInQueryNearestList[nearestIdVecs[j].first ]++;
        }

        for(int j = 0 ; j < tops4EachQueryTerm ; j++)
        {
            temp[j].first /= totalSc;

            idScore[temp[j].second] += temp[j].first;

        }



        //for(int ii = i * tops4EachQueryTerm ; ii < probWordVec.size() ; ii++)
        //probWordVec[ii].first /= totalSc;
        //cerr<<ind.term(queryTermsIdVec[i].first)<<" ";
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
    {
        double hscore = mit->second * (tidNumOfOccurInQueryNearestList[mit->first]);
        probWordVec.push_back(make_pair<double,int>(hscore, mit->first) );
    }


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)
    /*cerr<<" ::: ";
    for(int i = 0 ; i < probWordVec.size() ;i ++)
        cerr<< ind.term(probWordVec[i].second)<<" ";
    cerr << endl;*/

    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    for (int i = 0; i < tops4EachQuery; i++)//v
        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);


    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    delete fblm;


#endif

#if COMBSUM

    //cerr<<relJudgDoc.size()<<endl;
    //cerr<<"injaaaaaaaaaaaaa\n";
    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;

    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;

        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;
        nearestTerm2Vec(qTermVec ,relJudgDoc, nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;

            //probWordVec.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
        }

        for(int j = 0 ; j < nearestIdVecs.size()/*tops4EachQueryTerm*/ ; j++)
        {
            temp[j].first /= totalSc;

            double idf1 = log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first ) );
            double idf2= log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first) ) /(log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[i].first) )+1 ) ;
            double dist = weightedQueryTerms[i].second;
            double expdist = exp(dist);
            double impOfQTerm = 1.0;

            idScore[temp[j].second] += temp[j].first *impOfQTerm  ;
        }
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
        probWordVec.push_back(make_pair<double,int>(mit->second, mit->first) );


    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)


    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);


    //    cerr<<":::::\n";
    //    for(int i = 0 ;i < queryTermsIdVec.size();i++)
    //    {
    //        cerr<<ind.term(queryTermsIdVec[i].first)<<" ";
    //    }
    //    cerr<<"\n";
    //    for (int i = 0; i < tops4EachQuery; i++)//v
    //    {
    //        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);
    //        cerr<<ind.term(probWordVec[i].second)<<" "<<probWordVec[i].first<<", ";
    //    }
    //    cerr<<endl<<endl;



    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);



    delete fblm;
#endif
#if CENTROID
    //cosWithAvg(Centroid)

    map<int, vector<double> >::iterator endIt = wordEmbedding.end();

    vector<pair<double, int> >probWordVec;
    lemur::langmod::DocUnigramCounter *dCounter;

    //dCounter = new lemur::langmod::DocUnigramCounter(ind);//Collection
    dCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc, ind);//relJudDocs

    double total_score = 0.0;
    dCounter->startIteration();
    while(dCounter->hasMore())
    {
        int eventInd;
        double weight;
        dCounter->nextCount(eventInd,weight);

        map<int, vector<double> >::iterator tempit = wordEmbedding.find(eventInd);
        if( tempit != endIt )
        {
            vector<double> tt = tempit->second;
            //vector<double> tt = wordEmbedding[eventInd];

            //float sc = cosineSim(Vq , tt);
            float sc = softMaxFunc(Vq , tt);
            //float sc = softMaxFunc2(Vq , tt);
            //sc += 1.0;

            total_score += sc;
            probWordVec.push_back(pair<double,int>(sc,eventInd));

        }
    }
    for(int i = 0 ; i < probWordVec.size() ; i++)
        probWordVec[i].first /= total_score;

    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);



    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    int countPos = min((int)tops4EachQuery , (int)probWordVec.size());

    for (int i = 0; i < countPos; i++)
    {
        //cerr<<probWordVec[i].second;
        //cerr<<" "<<probWordVec[i].first<<" ";
        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);
    }


    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());

    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);


    delete dCounter;
    delete fblm;
    //delete qr;
#endif


#if LOGLOGISTIC

    //log-logistic
    vector<pair<double, int> >finalScoreIdVec;

    map<int , double> idProbMap;
    map<int , double >::iterator endMapIt = idProbMap.end();
    map<int, vector<double> >::iterator endIt = wordEmbedding.end();

    const double avgl = ind.docLengthAvg();
    const double N = ind.docCount();

    vector<pair<double, int> >selectedWordProbId;

    //ofstream write ("QueryWord-expanded-terms-2.txt", fstream::app);
    for(int ii = 0 ; ii < queryTermsIdVec.size() ; ii++)//qi
    {
        double totalScore = 0.0 ;
        finalScoreIdVec.clear();
        idProbMap.clear();



        for(int i = 0 ; i < relJudgDoc.size() ; i++)
        {
            lemur::langmod::DocUnigramCounter *myDocCounter;
            myDocCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc[i], ind);


            myDocCounter->startIteration();
            while(myDocCounter->hasMore())
            {
                int eventInd;
                double weight;
                myDocCounter->nextCount(eventInd,weight);

                map<int, vector<double> >::iterator tempit = wordEmbedding.find(eventInd);
                if( tempit != endIt )//found!
                {
                    vector<double> tt = tempit->second;

                    double sc = cosineSim(queryTermsIdVec[ii].second , tt);
                    //double sc = softMaxFunc2(queryTermsIdVec[ii].second , tt);
                    //double sc = softMaxFunc(queryTermsIdVec[ii].second , tt);

                    double TF = weight;
                    double docLength = ind.docLength( relJudgDoc[i] );
                    double DF = ind.docCount(eventInd);//df

                    double lambda_w = DF/N;
                    double tf_w = TF * log(1 + ((avgl)/docLength) ) ;
                    double score_ = log(( tf_w + lambda_w )/lambda_w );

                    score_ *= exp(sc); //CHECK IT!!! (+1 removed)// [-1:1]->[0:2]
                    score_ *= sc;


                    totalScore += score_;

                    map<int , double >::iterator fit = idProbMap.find(eventInd);
                    if(fit != endMapIt)
                    {
                        //cerr<<"before: "<<idProbMap[eventInd];
                        idProbMap[eventInd] += score_;
                        //cerr<<" after: "<<idProbMap[eventInd]<<endl;
                    }
                    else
                        idProbMap.insert(pair<int,double>(eventInd , score_));

                    //cerr<<TF<<" "<<docLength<<" "<<DF<<" "<<lambda_w<<" "<<tf_w<<" "<<score_<<endl;
                }

            }

            delete myDocCounter;
        }//end-for-reljudg

        //double idf1 = log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[ii].first ) );
        //double idf2= log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[ii].first) ) /(log((double)ind.docCount() / (double)ind.docCount(queryTermsIdVec[ii].first) )+1 ) ;
        //double dist = weightedQueryTerms[ii].second;
        //double expdist = exp(dist);
        double impOfQTerm = 1.0;

        for(map<int , double>::iterator it = idProbMap.begin() ; it != idProbMap.end() ; ++it )
        {
            finalScoreIdVec.push_back( make_pair<double,int>( (it->second * impOfQTerm )/totalScore , it->first) );
        }

        std::sort(finalScoreIdVec.begin() , finalScoreIdVec.end() , pairCompare);// can use top n selecting algorithm O(n)


        //write<<ind.term(queryTermsIdVec[ii].first)<<"-> "<<endl;


        int cc=-1;
        if( tops4EachQueryTerm < finalScoreIdVec.size())
            cc = tops4EachQueryTerm ;
        else
            cc = finalScoreIdVec.size();

        //int cc = finalScoreIdVec.size();
        for(int i = 0 ; i < cc ; i++)
        {
            selectedWordProbId.push_back(make_pair<double,int>(finalScoreIdVec[i].first, finalScoreIdVec[i].second) );
            //write <<"( "<<ind.term(finalScoreIdVec[i].second) << " "<< finalScoreIdVec[i].first <<") , ";
            //cerr<<"( "<<ind.term(finalScoreIdVec[i].second) << " "<< finalScoreIdVec[i].first <<") , ";
        }

        //write<<endl<<endl;
        //cerr<<totalScore<<" ";

    }//end-for-query
    //write.close();

    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    int countPos = selectedWordProbId.size();
    for (int i = 0; i < countPos; i++)
        lmCounter.incCount(selectedWordProbId[i].second , selectedWordProbId[i].first);

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    //delete dCounter;
    delete fblm;
    //delete qr;

#endif

#if COSREL

    //cosWithQueryTerms
    map<int, vector<double> >::iterator endIt = wordEmbedding.end();


    vector<pair<double, int> >selectedWordProbId;
    vector<pair<double, int> >probWordVec;
    lemur::langmod::DocUnigramCounter *dCounter;
    dCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc, ind);

    //ofstream write ("QueryWord-expanded-terms.txt", fstream::app);

    //cerr<<queryTermsIdVec.size()<<" "<<ind.term("marchandis")<<" ";
    for(int ii = 0 ; ii < queryTermsIdVec.size() ; ii++)
    {
        //cerr<< ind.term( queryTermsIdVec[ii].first ) <<" ";

        probWordVec.clear();

        dCounter->startIteration();
        while(dCounter->hasMore())
        {
            int eventInd;
            double weight;
            dCounter->nextCount(eventInd,weight);

            map<int, vector<double> >::iterator tempit = wordEmbedding.find(eventInd);
            if( tempit != endIt )
            {
                vector<double> tt = tempit->second;
                //vector<double> tt = wordEmbedding[eventInd];


                double sc = cosineSim(queryTermsIdVec[ii].second , tt);
                //float sc = softMaxFunc(Vq , tt);
                //float sc = softMaxFunc2(Vq , tt);

                //cerr<<"sc: "<<sc<<" "<<ind.term(eventInd)<<endl;

                sc += 1.0;//[-1:1]-->[0:2]
                probWordVec.push_back(pair<double,int>( sc , eventInd));// can be select from Coll !!!!
            }
        }

        double total_sc= 0.0;

        //cerr<<"\nQUERY_TERM: "<<ind.term(queryTermsIdVec[ii].first)<<endl;
        for(int i = 0 ; i < probWordVec.size() ; i++)
        {



            //int cc = dCounter->count(probWordVec[i].second );
            //double score_ = log(1+cc)* exp(2.0*probWordVec[i].first);
            double score_ = /*log((double)ind.docCount() / (double)ind.docCount(probWordVec[i].first))**/probWordVec[i].first;//pure cos score
            //score_+=1.0; //[-1:1]-->[0:2]
            /*if(ind.term(probWordVec[i].second) == ind.term(queryTermsIdVec[ii].first) )
                cerr<<ind.term(queryTermsIdVec[ii].first)<<" "<<cc<<" "<<probWordVec[i].first<<" "<<score_<<endl;
            if(ind.term(probWordVec[i].second) == "of" )
                cerr<<"of: "<<cc<<" "<<probWordVec[i].first<<" "<<score_<<endl;
            else if(ind.term(probWordVec[i].second) == "dope" )
                cerr<<"dope: "<<cc<<" "<<probWordVec[i].first<<" "<<score_<<endl;*/



            probWordVec[i].first =  score_ ;
            total_sc += score_;
        }
        for(int i = 0 ; i < probWordVec.size() ; i++)
        {
            probWordVec[i].first /= total_sc;

            /*if(ind.term(probWordVec[i].second) == ind.term(queryTermsIdVec[ii].first) )
            cerr<<ind.term(queryTermsIdVec[ii].first)<<" weight: "<<probWordVec[i].first<<endl;
        if(ind.term(probWordVec[i].second) == "dope" )
            cerr<<"dope weight: "<<probWordVec[i].first<<endl;
        else if(ind.term(probWordVec[i].second) == "of" )
            cerr<<"of weight: "<<probWordVec[i].first<<endl;
            */
        }
        std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)


        int cc = -1;
        if(tops4EachQueryTerm < probWordVec.size())
            cc = tops4EachQueryTerm;
        else
            cc = probWordVec.size();
        //int cc = probWordVec.size();
        //write<<ind.term(queryTermsIdVec[ii].first)<<"-> "<<endl;
        for(int i = 0 ; i < cc ; i++)
        {
            selectedWordProbId.push_back( probWordVec[i] );

            //write <<"( "<<ind.term(probWordVec[i].second) << " "<< probWordVec[i].first <<") , ";
        }

        //write<<endl<<endl;
    }//end-for
    //write.close();


    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    int countPos = selectedWordProbId.size();
    for (int i = 0; i < countPos; i++)
    {
        //cerr<<probWordVec[i].second;
        //cerr<<" "<<probWordVec[i].first<<" ";
        lmCounter.incCount(selectedWordProbId[i].second , selectedWordProbId[i].first);
    }

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);




    delete dCounter;
    delete fblm;
    //delete qr;
#endif
#if 0
    //cosWithAvg

    map<int, vector<double> >::iterator endIt = wordEmbedding.end();

    vector<pair<double, int> >probWordVec;
    lemur::langmod::DocUnigramCounter *dCounter;
    dCounter = new lemur::langmod::DocUnigramCounter(relJudgDoc, ind);

    dCounter->startIteration();
    while(dCounter->hasMore())
    {
        int eventInd;
        double weight;
        dCounter->nextCount(eventInd,weight);

        map<int, vector<double> >::iterator tempit = wordEmbedding.find(eventInd);
        if( tempit != endIt )
        {
            vector<double> tt = tempit->second;
            //vector<double> tt = wordEmbedding[eventInd];

            float sc = cosineSim(Vq , tt);
            //float sc = softMaxFunc(Vq , tt);
            //float sc = softMaxFunc2(Vq , tt);
            sc += 1.0;
            probWordVec.push_back(pair<double,int>(sc,eventInd));

        }
    }
    double total_sc= 0;
    for(int i = 0 ; i < probWordVec.size() ; i++)
    {
        int cc = dCounter->count(probWordVec[i].second );
        double score_ = log(1+cc)* exp(2*probWordVec[i].first);
        probWordVec[i].first =  score_ ;
        total_sc += score_;
    }
    for(int i = 0 ; i < probWordVec.size() ; i++)
        probWordVec[i].first /= total_sc;


    //std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);


    /*for(int i = 0 ; i < numberOfPositiveSelectedTopWord ;i ++)
        cerr << ind.term(probWordVec[i].second) <<" ";
    cerr<<endl;*/


    COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    //int countPos = min((int)numberOfPositiveSelectedTopWord , (int)probWordVec.size());
    /*int countPos = -1;
    if(numberOfPositiveSelectedTopWord < probWordVec.size() )
        countPos = numberOfPositiveSelectedTopWord;
    else
        countPos = probWordVec.size();*/
    int countPos = probWordVec.size();
    for (int i = 0; i < countPos; i++)
    {
        //cerr<<probWordVec[i].second;
        //cerr<<" "<<probWordVec[i].first<<" ";
        lmCounter.incCount(probWordVec[i].second , probWordVec[i].first);
    }

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);

    delete dCounter;
    delete fblm;
    //delete qr;
#endif

}


float lemur::retrieval::RetMethod::cosineSim(vector<double> aa, vector<double> bb)
{
    double numerator = 0.0 ,denominator =0.0,sum_a = 0.0,sum_b = 0.0;
    for(int i = 0 ; i< aa.size() ; i++)
    {
        double a = aa[i];
        double b = bb[i];
        numerator += a*b;
        sum_a += a*a;
        sum_b += b*b;
    }
    denominator = sqrt(sum_a) * sqrt(sum_b);
    //cout<< numerator/denominator;

    return (numerator /denominator) ;

}
float lemur::retrieval::RetMethod::softMaxFunc(vector<double> aa, vector<double> bb)
{
    return exp(cosineSim(aa,bb));
}
float lemur::retrieval::RetMethod::softMaxFunc2(vector<double> aa, vector<double> bb)
{
    return 1/1+exp(5.0*cosineSim(aa,bb));
}

vector<double> lemur::retrieval::RetMethod::extractKeyWord(int newDocId)
{
    double numberOfSelectedTopWord = 10.0;
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter;
    dCounter = new lemur::langmod::DocUnigramCounter(newDocId, ind);


    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];


    double meanLL=1e-40;
    double distQueryNorm=0;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm += distQueryEst[i];
    }

    double noisePr = 0.9; //qryParam.fbMixtureNoise;
    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (int i=1; i<=numTerms;i++)
        {
            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001)
        {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1  << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts
        dCounter->startIteration();
        while (dCounter->hasMore())
        {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (int i=1; i<=numTerms; i++)
        if (distQuery[i] > 0)
            lmCounter.incCount(i, distQuery[i]);


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    //origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
    //                      qryParam.fbPrSumTh, qryParam.fbPrTh);

    vector<pair<double, int> >probWordVec;

    fblm->startIteration();
    while(fblm->hasMore())
    {
        int wid=-10;
        double wprob=0.0;
        fblm->nextWordProb(wid,wprob);
        probWordVec.push_back(pair<double,int>(wprob,wid));
    }

    //ofstream outputfile;
    //outputfile.open("salamsalam");

    vector<double>avgDoc(W2VecDimSize ,0.0);

    std::sort(probWordVec.begin(),probWordVec.end(),pairCompare);

    double wordCount = std::min((double)probWordVec.size() , numberOfSelectedTopWord);

    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    for(int i = 0 ; i < wordCount; i++)
    {
        const std::map<int,vector<double> >::iterator it = wordEmbedding.find(probWordVec[i].second);
        if( it != endIt)
        {
            vector<double> tt = it->second ;
            for(int jj = 0 ;jj < W2VecDimSize ;jj++)
                avgDoc[jj] +=  tt[jj] ;

        }
        //else: cerr<<"fix me11!\n";//Fix Me!!!!!!!!!!!!

    }

    for(int i=0 ;i < W2VecDimSize ; i++)
    {
        //cout<<avgDoc[i]<<" ";
        avgDoc[i] /= wordCount;
        //cout<<avgDoc[i]<<"\n";
    }

    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

    return avgDoc;

}
float lemur::retrieval::RetMethod::computeProfDocSim(lemur::api::TextQueryRep *textQR,int docID ,
                                                     vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel , bool newRel)
{
#if 1
    double sc = scoreDoc(*textQR ,docID);
    //cerr<<sc<<" ";
    return sc;
#endif
#if 0
    vector<double> newDocAvgKeyWordsVec = docIdKeyWords[docID];//extractKeyWord(docID) ;//;
    return cosineSim(Vq,newDocAvgKeyWordsVec);

#endif
#if 0
    vector<vector<double> > docTerms;
    double counter = 0.0 ;

    TermInfoList *docTermInfoList =  ind.termInfoList(docID);
    docTermInfoList->startIteration();
    const std::map<int,vector<double> >::iterator endIt = wordEmbedding.end();
    while(docTermInfoList->hasMore())
    {

        TermInfo *ti = docTermInfoList->nextEntry();
        const std::map<int,vector<double> >::iterator itt = wordEmbedding.find(ti->termID());
        if(itt != endIt)
        {
            counter += 1;
            docTerms.push_back(itt->second);
        }
        else
            continue;

        //delete ti;
    }
    delete docTermInfoList;


    vector<double>docAvg (W2VecDimSize,0.0);
    for(int i =0 ; i< docTerms.size() ; i++)
    {
        for(int j = 0 ;j < W2VecDimSize/*docTerms[i].size()*/ ; j++)
            docAvg[j] += docTerms[i][j];
    }
    //cout<<docAvg[0]<<" "<<counter<<endl;
    for(int i = 0 ; i < docAvg.size() ;i++)
        docAvg[i] /= (double)(docTerms.size());
    //cout<<docAvg[0]<<endl;


    return cosineSim(Vq , docAvg);
#endif

#if 0
    IndexedRealVector nonRel,rel;
    for (int i =0 ; i<nonReljudgDoc.size() ; i++)
    {
        nonRel.PushValue(nonReljudgDoc[i],0);
    }
    PseudoFBDocs  *nonRelDocs;
    nonRelDocs= new PseudoFBDocs(nonRel,-1,true);

    for(int i =0 ; i< relJudgDoc.size();i++)
        rel.PushValue(relJudgDoc[i],0);
    PseudoFBDocs  *relDocs;
    relDocs= new PseudoFBDocs(rel,-1,true);

    const QueryModel *qm = dynamic_cast<const QueryModel *>(textQR);
    //cout<<"positive score"<<endl;
    double sc = 0;
    DocumentRep *dRep;
    HashFreqVector hfv(ind,docID);

    dRep = computeDocRep(docID);
    textQR->startIteration();
    while (textQR->hasMore())
    {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0)
        {
            cerr<<"**********"<<endl;
            //break;
            continue;
        }

        int tf;
        hfv.find(qTerm->id(),tf);
        DocInfo *info = new DocInfo(docID,tf);


        sc += scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep);//QL = sc+=|q|*log( p_seen(w|d)/(a(d)*p(w|C)) ) [slide7-11]
        //cout<<ind.term(qTerm->id())<<": "<<scoreFunc()->matchedTermWeight(qTerm, textQR, info, dRep)<<endl;
        delete info;
        delete qTerm;
    }

    double negQueryGenerationScore=0.0;
    //cout<<"negative score:"<<endl;
    if(RSMethodHM == 1)//RecSys(neg,coll)
    {
        negQueryGenerationScore= qm->negativeQueryGeneration(dRep ,nonReljudgDoc ,relJudgDoc,negGenModeHM, newNonRel,newRel,NegMu,delta,lambda_1,lambda_2);
    }
    else if (RSMethodHM == 2 || RSMethodHM == 3)//RecSys negKLQTE(2) and negKL(3)
    {
        negQueryGenerationScore = qm->negativeKL(dRep ,nonReljudgDoc , newNonRel,NegMu);
    }
    /*else if (RSMethodHM == 4)//fang
            {
                negQueryGenerationScore = fangScore(*nonRelDocs,docID,newNonRel);
            //    cout<<"inja5"<<endl;
            }*/

    //double fangScoreTmp = fangScore(*relDocs,docID,newRel);//considering positive feedback
    //negQueryGenerationScore -= fangScore(*relDocs,docID,newRel);//considering positive feedback


    double adjustedScore = scoreFunc()->adjustedScore(sc, textQR, dRep);
    //cout<<"negqueryScore: "<<negQueryGenerationScore<<endl;

    //cout<<"fangScoreTmp: "<< fangScoreTmp<<" negqueryScore: "<<negQueryGenerationScore<<" adjusted: "<<adjustedScore<<" newRel" <<newRel<<endl;
    //negQueryGenerationScore -= fangScoreTmp;

    //cout <<"inja6"<<endl;
    delete dRep;
    delete nonRelDocs;
    delete relDocs;
    // cout<<"neg score:**********:"<<negQueryGenerationScore<<endl;
    //cout<<"pos score:**********:"<<adjustedScore<<endl;
    return (negQueryGenerationScore + adjustedScore);

#endif

}


void lemur::retrieval::RetMethod::updateTextQuery(TextQueryRep &origRep,
                                                  const DocIDSet &relDocs,const DocIDSet &nonRelDocs )
{
    //cout<<"RM: "<<RM<<endl;
    QueryModel *qr;

    qr = dynamic_cast<QueryModel *> (&origRep);

    if(RM=="RM1"){
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="RM2"){
        computeRM2FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM3"){
        computeRM3FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM4"){
        computeRM4FBModel(*qr, relDocs);
        return;
    }else if(RM=="MIX"){
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="DIVMIN"){
        computeDivMinFBModel(*qr, relDocs);
        return;
    }else if(RM=="MEDMM"){
        computeMEDMMFBModel(*qr, relDocs);
        return;
    }




    switch (qryParam.fbMethod) {
    case RetParameter::MIXTURE:
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::DIVMIN:
        computeDivMinFBModel(*qr, relDocs);
        break;
    case RetParameter::MARKOVCHAIN:
        computeMarkovChainFBModel(*qr, relDocs);
        break;
    case RetParameter::RM1:
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::RM2:
        computeRM2FBModel(*qr, relDocs);
        break;
    default:
        throw Exception("SimpleKLRetMethod", "unknown feedback method");
        break;
    }
}


void lemur::retrieval::RetMethod::computeNearestTerm2Vec(vector<double> vec )
{

    ofstream inputfile;
    inputfile.open("outputfiles/NearestTerm2Vec.txt",ios::app);

    vector<pair<double,int> >simTermid;

    for(int i =1 ; i< ind.termCountUnique() ; i++)
    {

        vector<double>dtemp;
        if(wordEmbedding.find(i) != wordEmbedding.end())
        {
            dtemp.assign(wordEmbedding[i].begin() ,wordEmbedding[i].end());
        }
        else
        {
            continue;
        }

        double sim = this->cosineSim(vec,dtemp);
        simTermid.push_back(pair<double,int>(sim,i));
    }
    std::sort(simTermid.begin() , simTermid.end(),pairCompare);


    for(int i = 0 ; i < 15 ; i++)
        inputfile <<"( "<< ind.term(simTermid[i].second)<<" , "<<simTermid[i].first<<" ) ";

    inputfile<<endl<<endl;
    inputfile.close();

}


void lemur::retrieval::RetMethod::updateThreshold(lemur::api::TextQueryRep &origRep,
                                                  vector<int> relJudgDoc ,vector<int> nonRelJudgDoc , int mode)
{

    if(mode == 0)//non rel passed
    {
        setThreshold(getThreshold()+getC1());
    }
    else //if(mode == 1)//not showed anything
    {
        setThreshold( getThreshold()- getC2() );
    }

}

void lemur::retrieval::RetMethod::computeMixtureFBModel(QueryModel &origRep,
                                                        const DocIDSet &relDocs, const DocIDSet &nonRelDocs )
{
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];

    double noisePr;

    int i;

    double meanLL=1e-40;
    double distQueryNorm=0;

    for (i=1; i<=numTerms;i++) {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }
    noisePr = qryParam.fbMixtureNoise;

    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (i=1; i<=numTerms;i++) {

            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001) {
            //cerr << "converged at "<< qryParam.emIterations - itNum+1
            //   << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts

        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}


void lemur::retrieval::RetMethod::computeDivMinFBModel(QueryModel &origRep,
                                                       const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    double * ct = new double[numTerms+1];

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    relDocs.startIteration();
    while (relDocs.hasMore()) {
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));

        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += log(dm->unseenCoeff()*collectLM->prob(i));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += log(dm->seenProb(info->count(), info->termID())/
                                      (dm->unseenCoeff()*collectLM->prob(info->termID())));
        }
        delete tList;
        delete dm;
    }
    if (actualDocCount==0) return;

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    double norm = 1.0/(double)actualDocCount;
    for (i=1; i<=numTerms; i++) {
        lmCounter.incCount(i,
                           exp((ct[i]*norm -
                                qryParam.fbMixtureNoise*log(collectLM->prob(i)))
                               / (1.0-qryParam.fbMixtureNoise)));
    }
    delete [] ct;


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMEDMMFBModel(QueryModel &origRep,
                                                      const DocIDSet &relDocs)
{
    //cout<<"MEDMM\n";
    COUNT_T numTerms = ind.termCountUnique();
    set <int> fbterms;
    double * ct = new double[numTerms+1];
    vector<pair<int,double> > scores;
    //qryParam.fbCoeff =fbcoef;

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    int Vf = 0;
    relDocs.startIteration();
    while (relDocs.hasMore()){
        int id;
        double pr;
        relDocs.nextIDInfo(id, pr);
        TermInfoList *tList = ind.termInfoList(id);
        Vf += tList->size();
        delete tList;
    }


    relDocs.startIteration();
    double sum_alpha = 0;
    vector <double> alpha_d;
    while (relDocs.hasMore()) {
        alpha_d.push_back(1);
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));


        origRep.startIteration();
        while (origRep.hasMore()) {
            QueryTerm *qt = origRep.nextTerm();
            TermInfo *info;
            TermInfoList *tList = ind.termInfoList(id);
            tList->startIteration();
            bool flag = true;
            while (tList->hasMore()) {
                info = tList->nextEntry();
                if (info->termID() == qt->id()){
                    alpha_d[alpha_d.size()-1] *= (info->count()+0.1)/(ind.docLength(id)+0.1*Vf);
                    flag = false;
                    break;
                }
            }
            if (flag){
                alpha_d[alpha_d.size()-1] *= (0.1)/(ind.docLength(id)+0.1*Vf);
            }
            delete tList;
            delete qt;
        }
        sum_alpha += (alpha_d[alpha_d.size()-1]);
        delete dm;
    }
    relDocs.startIteration();
    int doc_id = 0;
    while (relDocs.hasMore()){
        int id;
        double pr;
        relDocs.nextIDInfo(id, pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel*> (computeDocRep(id));
        //cerr<<"++++"<<exp(alpha_d[doc_id])/sum_alpha<<endl;
        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += (((alpha_d[doc_id])/sum_alpha)*log(dm->unseenCoeff()*collectLM->prob(i)));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += (((alpha_d[doc_id])/sum_alpha)*log(((info->count()+0.1)/(ind.docLength(id)+0.1*Vf))/
                                                                     (dm->unseenCoeff()*collectLM->prob(info->termID()))));
            fbterms.insert(info->termID());
        }
        delete tList;
        delete dm;
        doc_id++;
    }
    if (actualDocCount==0) return;
    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    double beta =1.2;
    double lambda = 0.1;
    double norm = 1.0/(double)actualDocCount;

    for (i=1; i<=numTerms; i++) {
        if(exp((1.0/beta)*(ct[i] -lambda*log(collectLM->prob(i))))>0){
            scores.push_back(make_pair(i,exp((1.0/beta)*(ct[i] - lambda*log(collectLM->prob(i)))) ));
            lmCounter.incCount(i,
                               exp((1.0/beta)*(ct[i] -
                                               lambda*log(collectLM->prob(i)))));
        }
    }
    delete [] ct;
    std::sort(scores.begin(),scores.end(),sort_pred());

    ofstream write ("expanded-terms.txt", fstream::app);
    for(int i=0; i<qryParam.fbTermCount;i++) {
        write<<ind.term(scores[i].first)<<":"<<scores[i].second<<" ";
    }
    write<<endl;
    write.close();



    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMarkovChainFBModel(QueryModel &origRep, const DocIDSet &relDocs)
{
    int stopWordCutoff =50;

    lemur::utility::ArrayCounter<double> *counter = new lemur::utility::ArrayCounter<double>(ind.termCountUnique()+1);

    lemur::langmod::OneStepMarkovChain * mc = new lemur::langmod::OneStepMarkovChain(relDocs, ind, mcNorm,
                                                                                     1-qryParam.fbMixtureNoise);
    origRep.startIteration();
    double summ;
    while (origRep.hasMore()) {
        QueryTerm *qt;
        qt = origRep.nextTerm();
        summ =0;
        mc->startFromWordIteration(qt->id());
        // cout << " +++++++++ "<< ind.term(qt->id()) <<endl;
        TERMID_T fromWd;
        double fromWdPr;

        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }
            summ += qt->weight()*fromWdPr*collectLM->prob(fromWd);
            // summ += qt->weight()*fromWdPr;
        }
        if (summ==0) {
            // query term doesn't exist in the feedback documents, skip
            continue;
        }

        mc->startFromWordIteration(qt->id());
        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }

            counter->incCount(fromWd,
                              (qt->weight()*fromWdPr*collectLM->prob(fromWd)/summ));
            // counter->incCount(fromWd, (qt->weight()*fromWdPr/summ));

        }
        delete qt;
    }
    delete mc;

    lemur::langmod::UnigramLM *fbLM = new lemur::langmod::MLUnigramLM(*counter, ind.termLexiconID());

    origRep.interpolateWith(*fbLM, 1-qryParam.fbCoeff, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fbLM;
    delete counter;
}


void lemur::retrieval::RetMethod::computeRM3FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();
    //qryParam.fbCoeff =fbcoef;

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double expWeight = qryParam.fbCoeff;

    TERMID_T i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }

    for (i=1; i<=numTerms;i++) {
        distQuery[i] = (expWeight*distQuery[i]/pSum +
                        (1-expWeight)*ind.termCount(i)/ind.termCount());
    }


    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    //origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
    //		qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
}


// out: w.weight = P(w|Q)
// P(w|Q) = k P(w) P(Q|w)
// P(Q|w) = PROD_q P(q|w)
// P(q|w) = SUM_d P(q|d) P(w|d) p(d) / p(w)
// P(w) = SUM_d P(w|d) p(d)
// Promote this to some include somewhere...
struct termProb  {
    TERMID_T id; // TERM_ID
    double prob; // a*tf(w,d)/|d| +(1-a)*tf(w,C)/|C|
};
#if 0
//RM1
COUNT_T numTerms = ind.termCountUnique();

// RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

double *distQuery = new double[numTerms+1];
double expWeight = qryParam.fbCoeff;

TERMID_T i;
for (i=1; i<=numTerms;i++)
distQuery[i] = 0.0;

double pSum=0.0;
dCounter->startIteration();
while (dCounter->hasMore()) {
    int wd; // dmf FIXME
    double wdPr;
    dCounter->nextCount(wd, wdPr);
    distQuery[wd]=wdPr;
    pSum += wdPr;
}

for (i=1; i<=numTerms;i++) {
    distQuery[i] = expWeight*distQuery[i]/pSum +
            (1-expWeight)*ind.termCount(i)/ind.termCount();
}

lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
for (i=1; i<=numTerms; i++) {
    if (distQuery[i] > 0) {
        lmCounter.incCount(i, distQuery[i]);
    }
}

/***/
vector<pair<double ,int> >probIdVec;
lmCounter.startIteration();
while(lmCounter.hasMore())
{
    int id;
    double wdPr;
    lmCounter.nextCount(id,wdPr);
    probIdVec.push_back(make_pair<double,int>(wdPr , id));
}
std::sort(probIdVec.begin(), probIdVec.end() ,pairCompare);

double totalSc = 0;
for(int i = 0 ; i < topsCinRM1 ;i++)
totalSc += probIdVec[i].first;

lemur::utility::ArrayCounter<double> myLmCounter(numTerms+1);
for(int i = 0 ; i < topsCinRM1 ;i++)
myLmCounter.incCount(probIdVec[i].second, (probIdVec[i].first/totalSc) );

/***/

//lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(myLmCounter, ind.termLexiconID());
//origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
//        qryParam.fbPrSumTh, 0.0);
//delete fblm;

delete dCounter;
delete[] distQuery;
///end compute p_clip(t|RM1)


#endif


void lemur::retrieval::RetMethod::computeRM2FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs) {
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0;
                P_q_d=0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;

                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] =P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

void lemur::retrieval::RetMethod::computeRM4FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    //qryParam.fbCoeff =fbcoef;
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }
    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }
    //for(int i=0;i<numTerms+1;i++)
    //distQuery[i]=distQuery[i]/pSum;

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        P_w=distQuery[wd];
        //cerr << P_w << endl;
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0.0;
                P_q_d=0.0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;
                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                //cerr<<"###########"<<P_q_d<<" "<<P_w_d<<endl;
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] = P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        //if (distQuery[i] > 0) {
        lmCounter.incCount(i, distQuery[i]);
        //}
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 1.0-qryParam.fbCoeff, qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

#if 0
void lemur::retrieval::RetMethod::computeWEFBModel(QueryModel &origRep,const DocIDSet &relDocs)
{

    vector<double> q2v;
    double eps = 0.0000000001;
    double sum=eps;
    map<int,double> feedback_w2v_model;
    map<int,int> word_count;
    int tot_count=0;
    int feedback_voc_count=0;
    vector<int> VS;
    COUNT_T numTerms = ind.termCountUnique();
    double distQueryNorm;
    double noisePr = qryParam.fbMixtureNoise;
    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];
    qryParam.fbCoeff =fbcoef;

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    vector<pair<int,double> > scores;
    vector<pair<int,double> > pos;
    vector<pair<int,double> > neg;
    set<int> pos_terms;
    set<int> neg_terms;

    for (int i=1; i<=numTerms;i++)
    {
        distQueryEst[i] = 0.0;//rand()+0.001;
        distQuery[i] = 0.0;//rand()+0.001;
    }


    genSample(pos,neg,VS,word_count,relDocs,origRep);

    for(map<int,int>::iterator it = word_count.begin();it!=word_count.end();it++)
    {
        tot_count +=it->second;
    }
    //cerr<<tot_count<<endl;

    int i;
    int qcnt=0;
    q2v.resize(dim);
    for(int i=0;i<dim;i++){
        q2v[i] = 0.0;
    }

    origRep.startIteration();
    while (origRep.hasMore())
    {
        QueryTerm *qt = origRep.nextTerm();
        int id = qt->id();
        if(w2v[id].size()==0)
        {
            continue;
        }
        for(int i=0;i<dim;i++)
        {
            q2v[i] += w2v[id][i];
        }
        qcnt++;
        //cerr<<ind.term(id)<<endl;
        delete qt;
    }

    for(int i=0;i<dim;i++)
    {
        q2v[i] /= qcnt;
    }

    long double W[dim][dim]; // Feedback Matrix
    long double WT[dim][dim]; // Transformation Matrix
    long double delta_W[dim][dim];
    double etha = 0.0001;

    double alpha =we_alpha;
    double lambda =we_lambda;
    double beta =we_beta;

    int n_iter = 500;
    long double change_prev = 100.0;
    long double change_now = 1.0;
    for(int i=0;i<dim;i++)
    {
        for(int j=0;j<dim;j++)
        {
            double f = (double) rand()/RAND_MAX*1-0.5;
            W[i][j] = 0.f;
        }
    }

    while(abs(change_now)>0.000001 && n_iter != 0)
    {
        change_prev = change_now;
        change_now = 0.0;
        etha = 0.0001; //* (1+n_iter);
        //etha = 0.000001 * (n_iter);
        n_iter--;

        for(int x=0;x<dim;x++)
        {
            for(int y=0;y<dim;y++)
            {
                WT[x][y] =W[y][x];
                delta_W[x][y] = 0.f;
            }
        }

        // positive samples
        for(int i=0;i<pos.size();i++)
        {
            int wn = pos[i].first;
            pos_terms.insert(wn);
            double alpha_w = pos[i].second;
            long double Wvq[dim];
            double Wvq_vwn[dim];
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*q2v[y];
                }
                Wvq_vwn[x] = Wvq[x]-w2v[wn][x];
            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    delta_W[x][y] += etha*(alpha*Wvq_vwn[x]*q2v[y]);
                    W[x][y] -= etha*(alpha*Wvq_vwn[x]*q2v[y]); //FIXME
                }
            }
        }

        // negative samples
        for(int i=0;i<neg.size();i++)
        {
            int wn_ = neg[i].first;
            neg_terms.insert(wn_);
            double lambda_w = neg[i].second;
            long double Wvq[dim];
            double Wvq_vwn_[dim];
            for(int x=0;x<dim;x++)
            {
                Wvq[x]=0.0;
                for(int y=0;y<dim;y++)
                {
                    Wvq[x] +=WT[x][y]*q2v[y];
                }
                Wvq_vwn_[x] = Wvq[x]-w2v[wn_][x];
            }
            for(int x=0;x<dim;x++)
            {
                for(int y=0;y<dim;y++)
                {
                    delta_W[x][y] += etha*(-lambda*Wvq_vwn_[x]*q2v[y]);
                    W[x][y] -= etha*(-lambda*Wvq_vwn_[x]*q2v[y]);// FIXME
                }
            }
        }

        // regularization
        for (int x=0; x < dim; x++)
            for (int y = 0; y < dim; y++)
            {
                delta_W[x][y] -= etha*(beta*W[x][y]);
                W[x][y] -= -etha*(beta*W[x][y]); // FIXME
            }

        // update W
        for (int x = 0; x < dim; x++){
            for (int y = 0; y < dim; y++){
                //W[x][y] -= delta_W[x][y];//FIXME
                change_now += pow(delta_W[x][y],2);
            }
        }
        change_now = sqrt(change_now);//pow(change_now,1.0/4o);
        //cerr<<" # "<<change_now<<" ";
    }
    //cerr<<"(O o)"<<endl;

    sum=0.0;
    vector<double> WTvq;

    long double mu[dim]; // Feedback Matrix
    WTvq.resize(dim);
    for(int x=0;x<dim;x++)
    {
        double uv = 0;
        for(int y=0;y<dim;y++)
        {
            uv +=WT[x][y]*q2v[y] ;
        }
        WTvq[x]= uv;
    }

    for(int id=0;id<VS.size();id++)
    {
        vector<double> vec = w2v[VS[id]];
        int wn = VS[id];
        if(vec.size()==0){
            continue;
        }
        double sim =0.0;
        double euc =0.0;
        double norm1=0.0;
        for(int i=0;i<dim;i++)
        {
            euc += pow(vec[i]-WTvq[i],2);
            //euc += pow(vec[i]-q2v[i],2);
        }
        for(int i=0;i<dim;i++)
        {
            norm1+= pow(vec[i],2);
        }

        double norm2=0.0;
        for(int i=0;i<dim;i++)
        {
            norm2+= pow(WTvq[i],2);
            //norm2+= pow(q2v[i],2);
        }

        for(int i=0;i<dim;i++)
        {
            sim += vec[i]*WTvq[i];
            //sim += vec[i]*q2v[i];
        }

        //double prTopic =log(6.0/sqrt(2.0*M_PI)*exp(-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2)/0.1)) ;//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic =log(1.0+-pow((double)sim/(sqrt(norm1)*sqrt(norm2)),2));//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = sqrt(1+word_count[wn])*exp(1/(1+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = exp(1.0/(1.0+exp(-sim)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        double prTopic = word_count[wn]*exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //double prTopic = exp(sim/(sqrt(norm1)*sqrt(norm2)));//sqrt(euc);//(1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))/
        //((1-noisePr)*sim/(sqrt(norm1)*sqrt(norm2))+noisePr*collectLM->prob(wn));
        double p_w_f = (double)word_count[wn]/(double)tot_count;
        long double s = (double)sim/(sqrt(norm1)*sqrt(norm2));//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))
        //long double s =((1.0-noisePr)*p_w_f/((1-noisePr)*(p_w_f)+noisePr*collectLM->prob(wn)))*prTopic;//- 0.1 * (collectLM->prob(wn)));//*exp(prTopic);//(log(1.0+sim/(sqrt(norm1)*sqrt(norm2))))
        //cerr<<s<<endl;
        if(pos_terms.find(wn)!=pos_terms.end()){
            //if(neg_terms.find(wn)==neg_terms.end()){
            scores.push_back(make_pair(wn,prTopic));
            distQuery[wn] += prTopic;
            sum+=distQuery[wn];
        }
    }

    std::sort(scores.begin(),scores.end(),sort_pred());

    for(int i=0;i<scores.size();i++){
        int wn;
        scores[i].second /=sum;
        distQuery[wn] /=sum;
    }
    //std::sort(scores.begin(),scores.end(),sort_pred());
    ofstream write ("expanded-terms.txt", fstream::app);
    //for (int i=0; i<qryParam.fbTermCount;i++) {
    for (int i=0; i<scores.size();i++) {
        //for (int i=0; i<100;i++) {
        //write<<ind.term(scores[i].first)<<":"<<scores[i].second<<" ";
    }
    write<<endl;
    write.close();


    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    dCounter->startIteration();
    double counter =0;
    while (dCounter->hasMore()){
        int wd; //dmf FIXME
        double wdCt;
        dCounter->nextCount(wd, wdCt);
        counter+=distQuery[wd];
    }

    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;
}
#endif
void lemur::retrieval::RetMethod::computeRM1FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs,const DocIDSet &nonRelDocs)
{
#if 0
#define RM1SUM 1
#define RM1MNZ 0
#define RM1MAX 0


    ///compute p_clip(t|RM1)
    ///

    COUNT_T numTerms = ind.termCountUnique();

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    //double expWeight = qryParam.fbCoeff;

    TERMID_T i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }

    for (i=1; i<=numTerms;i++) {
        distQuery[i] = expWeight*distQuery[i]/pSum +
                (1-expWeight)*ind.termCount(i)/ind.termCount();
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }

    /***/
    vector<pair<double ,int> >probIdVec;
    lmCounter.startIteration();
    while(lmCounter.hasMore())
    {
        int id;
        double wdPr;
        lmCounter.nextCount(id,wdPr);
        probIdVec.push_back(make_pair<double,int>(wdPr , id));
    }
    std::sort(probIdVec.begin(), probIdVec.end() ,pairCompare);

    double totalSc = 0;
    for(int i = 0 ; i < topsCinRM1 ;i++)
        totalSc += probIdVec[i].first;

    lemur::utility::ArrayCounter<double> myLmCounter(numTerms+1);
    for(int i = 0 ; i < topsCinRM1 ;i++)
        myLmCounter.incCount(probIdVec[i].second, (probIdVec[i].first/totalSc) );

    /***/

    //lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(myLmCounter, ind.termLexiconID());
    //origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
    //        qryParam.fbPrSumTh, 0.0);
    //delete fblm;

    delete dCounter;
    delete[] distQuery;
    ///end compute p_clip(t|RM1)


#if RM1SUM
    /// compute p(.|M)
    vector<pair<double, int> >probWordVec;
    map<int, double> idScore;
    for(int i = 0 ; i < queryTermsIdVec.size() ; i++)
    {
        vector<pair<double, int> >temp;
        vector<double> qTermVec(queryTermsIdVec[i].second);
        vector<pair<int ,double> > nearestIdVecs;

        nearestTerm2Vec(qTermVec , nearestIdVecs);//n(50,100) most similar terms

        double totalSc =0;
        for(int j = 0 ;j < nearestIdVecs.size() ; j++)
        {
            double sc = exp(nearestIdVecs[j].second);
            totalSc+=sc;
            temp.push_back(make_pair<double, int> (sc, nearestIdVecs[j].first ) );
        }
        for(int j = 0 ; j < tops4EachQueryTerm ; j++)// hamun topsCinRM1 hast
        {
            temp[j].first /= totalSc;
            idScore[temp[j].second] += temp[j].first;
        }
    }
    map<int, double>::iterator mit ;
    for(mit = idScore.begin() ; mit != idScore.end() ;++mit)
    {
        probWordVec.push_back(make_pair<double,int>(mit->second, mit->first) );
    }

    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)

    //COUNT_T numTerms = ind.termCountUnique();
    lemur::utility::ArrayCounter<double> welmCounter(numTerms+1);

    for (int i = 0; i < tops4EachQuery; i++)//v
        welmCounter.incCount(probWordVec[i].second , probWordVec[i].first);

    //QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    //lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(welmCounter, ind.termLexiconID());
    //qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
#endif
    ///interpolate

    for (i=1; i<=numTerms; i++)
    {
        double prob =  myLmCounter.count(i) * alphaCoef + (1-alphaCoef) * welmCounter.count(i);
        probWordVec.push_back(make_pair<double, int>(prob,i));
    }
    std::sort(probWordVec.begin() , probWordVec.end() , pairCompare);// can use top n selecting algorithm O(n)

    lemur::utility::ArrayCounter<double> interpolateLmCounter(numTerms+1);
    for(int i = 0 ; i< tops4EachQuery ; i++)
    {
        interpolateLmCounter.incCount(probWordVec[i].second , probWordVec[i].first);
    }

    QueryModel *qr = dynamic_cast<QueryModel *> (&origRep);
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(interpolateLmCounter, ind.termLexiconID());
    qr->interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount, qryParam.fbPrSumTh, qryParam.fbPrTh);
    ///end compute p(.|M)
#endif
}
