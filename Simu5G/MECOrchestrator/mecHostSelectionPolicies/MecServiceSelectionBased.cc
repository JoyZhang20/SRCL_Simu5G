//
//                  Simu5G
//
// Authors: Giovanni Nardini, Giovanni Stea, Antonio Virdis (University of Pisa)
//
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself,
// and cannot be removed from it.
//

#include "nodes/mec/MECOrchestrator/mecHostSelectionPolicies/MecServiceSelectionBased.h"
#include "nodes/mec/MECPlatformManager/MecPlatformManager.h"
#include "nodes/mec/VirtualisationInfrastructureManager/VirtualisationInfrastructureManager.h"

using namespace std;
namespace simu5g {

cModule* MecServiceSelectionBased::findBestMecHost(const ApplicationDescriptor& appDesc)
{
    EV << "MecServiceSelectionBased::findBestMecHost - finding best MecHost..." << endl;
    cModule* bestHost = nullptr;
    bool found = false;

    // 读取当前时隙以及当前时隙决策序号
    int slot;  // 当前时隙
    int decisionNumber;  // 当前时隙决策序号
    std::ifstream slotFile("D:/Simu5G/slot.txt");
    if(slotFile){
        slotFile >> slot;
        slotFile.close();
    }else{
        EV << "can not read the slotFile" << endl;
    }
    std::ifstream decisionNumberFile("D:/Simu5G/decisionNumber.txt");
    if(decisionNumberFile){
        decisionNumberFile >> decisionNumber;
        decisionNumberFile.close();
    }else{
        EV << "can not read the decisionNumberFile" << endl;
    }

    // 2、读取任务数据以及决策
    std::ifstream decisionFile("D:/Simu5G/decision.txt");
    std::string line;
    vector<vector<int>> decisionData;
    if(decisionFile){
        // 读取 decision.txt 中的每一行，并将其转换为整数数组
        while (getline(decisionFile, line)) {
            std::stringstream ss(line);
            vector<int> row;
            string number;
            while (getline(ss, number, ',')) {
                row.push_back(stoi(number));
            }
            decisionData.push_back(row);
        }
        decisionFile.close();
    }else{
        EV << "can not read the decisionFile" << endl;
    }

    int currentDecision = decisionData[slot][decisionNumber];  // 获取当前决策


    for(auto mecHost : mecOrchestrator_->mecHosts)
    {
       //  our method
//       if(currentDecision == 0){
//           if (!strcmp(mecHost->getName(), "mecHost2")){
//               continue;
//           }
//           if (!strcmp(mecHost->getName(), "mecHost1")){
//               bestHost = mecHost;
//               break;
//           }
//       }else if(currentDecision == 1){
//           if (!strcmp(mecHost->getName(), "mecHost1")){
//              continue;
//           }
//           if (!strcmp(mecHost->getName(), "mecHost2")){
//              bestHost = mecHost;
//              break;
//           }
//       }

       EV << "MecServiceSelectionBased::findBestMecHost - MEC host ["<< mecHost->getName() << "] size of mecHost " <<  mecOrchestrator_->mecHosts.size() << endl;
       VirtualisationInfrastructureManager *vim = check_and_cast<VirtualisationInfrastructureManager*> (mecHost->getSubmodule("vim"));
       ResourceDescriptor resources = appDesc.getVirtualResources();
       bool res = vim->isAllocable(resources.ram, resources.disk, resources.cpu);
       if(!res)
       {
           EV << "MecServiceSelectionBased::findBestMecHost - MEC host ["<< mecHost->getName() << "] has not got enough resources. Searching again..." << endl;
           continue;
       }

       // Temporally select this mec host as the best
       EV << "MecServiceSelectionBased::findBestMecHost - MEC host ["<< mecHost->getName() << "] temporally chosen as bet MEC host, checking for the required MEC services.." << endl;
       bestHost = mecHost;

       MecPlatformManager *mecpm = check_and_cast<MecPlatformManager*> (mecHost->getSubmodule("mecPlatformManager"));
       auto mecServices = mecpm ->getAvailableMecServices();
       std::string serviceName;

       /* I assume the app requires only one mec service */
       if(appDesc.getAppServicesRequired().size() > 0)
       {
           serviceName =  appDesc.getAppServicesRequired()[0];
           EV << "MecServiceSelectionBased::findBestMecHost - required Mec Service: " << serviceName << endl;
       }
       else
       {
           EV << "MecServiceSelectionBased::findBestMecHost - the Mec App does not require any MEC service. Choose the temporary Mec Host as the best one"<< endl;
           found = true;
           break;
       }
       auto it = mecServices->begin();
       for(; it != mecServices->end() ; ++it)
       {
           if(serviceName.compare(it->getName()) == 0 && it->getMecHost().compare(bestHost->getName()) == 0)
           {
              EV << "MecServiceSelectionBased::findBestMecHost - The temporary Mec Host has the MEC service "<< it->getName() << " required by the Mec App. It has been chosen as the best one"<< endl;
              bestHost = mecHost;
              found = true;
              break;
           }
       }
       if(found)
           break;
    }

    // our method:记录新的时隙
//    decisionNumber++;
//    if(decisionNumber % 10 == 0){  // 如果一个时隙的决策全部读取完，则时隙数加1
//        slot++;
//        std::ofstream slotFileOut("D:/Simu5G/slot.txt");
//        if(slotFileOut){
//            slotFileOut << slot;
//            slotFileOut.close();
//        }else{
//            EV <<"can not write the slotFlie" << endl;
//        }
//    }
//    std::ofstream decisionNumberFileOut("D:/Simu5G/decisionNumber.txt");
//    if(decisionNumberFileOut){
//        decisionNumberFileOut << decisionNumber;
//        decisionNumberFileOut.close();
//    }else{
//        EV <<"can not write the decisionNumberFlie" << endl;
//    }

    if(bestHost != nullptr && !found)
       EV << "MecServiceSelectionBased::findBestMecHost - The best Mec Host hasn't got the required service. Best MEC host: " << bestHost << endl;
    else if(bestHost == nullptr)
       EV << "MecServiceSelectionBased::findBestMecHost - no MEC host found"<< endl;

    return bestHost;

}

} //namespace

