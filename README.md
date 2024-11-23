# SR-CL: Mobility-aware Seamless Service Migration and Resource Allocation in Multi-edge IoV Systems

Simu5G simulation code for implementing SR-CL

Note: This code cannot be run alone. You need to install the Simu5G simulation system and overwrite the code we provide to the original code to run it.

### Version

win11

OMNet++6.1.0

INET4.5.2

Simu5G

### Install OMNet++6.1.0

1. Download the installation package from [OMNet++ official website](https://omnetpp.org/download/) and unzip it locally

2. Execute mingwenv.cmd

3. Enter ./configure in the command line

4. Enter make -j8

### Install INET4.5.2

1. Create a directory as the workspace for OMNET++

2. Download the installation package from [INET Framework official website](https://inet.omnetpp.org/Download.html) and unzip it to the workspace in step 1

3. Change the folder name to inet4.5

4. Open OMNeT++ 6.1 IDE, File-Import-Existing Projects into Select the INET decompression directory in step 2 in Workspace

5. Fix bug
"../workspace/inet-4.5.2/src/inet/common/lifecycle/NodeStatus.cc":

```c++
Register_Enum(inet::NodeStatus, (NodeStatus::UP, NodeStatus::DOWN, NodeStatus::GOING_UP, NodeStatus::GOING_DOWN));
```

Change to

```c++
Register_Enum(inet::NodeStatus::State, (inet::NodeStatus::UP, inet::NodeStatus::DOWN, inet::NodeStatus::GOING_UP, inet::NodeStatus::GOING_DOWN));
```

"../workspace/inet-4.5.2/src/inet/linklayer/common/QosClassifier.ned": delete @enum()

6.Project-Build All (Ctrl+B)

### Install Simu5G

1. Download the code from [Simu5G repository](https://inet.omnetpp.org/Download.html) and unzip it to the workspace above

2. Open OMNeT++ 6.1 IDE, File-Import-Existing Projects into Workspace, select the Simu5G unzip directory in step 1 above

3.Project-Build All (Ctrl+B)

4. Fix bug: "../workspace/Simu5G/src/stack/pdcp_rrc/LtePdcpRrcBase.ned"

```c++
int conversationalRlc @enum(TM,UM,AM,UNKNOWN_RLC_TYPE) = default(1);
int streamingRlc @enum(TM,UM,AM,UNKNOWN_RLC_TYPE) = default(1);
int interactiveRlc @enum(TM,UM,AM,UNKNOWN_RLC_TYPE) = default(1);
int backgroundRlc @enum(TM,UM,AM,UNKNOWN_RLC_TYPE) = default(1);
```

Change to

``` c++
int TM = 0;
int UM = 1;
int AM = 2;
int UNKNOWN_RLC_TYPE = -1;
int conversationalRlc = default(UM);
int streamingRlc = default(UM);
int interactiveRlc = default(UM);
int backgroundRlc = default(UM);
```

### Verification

After opening Simu5G/simulations/NR/mec/requestResponseApp/omnetpp.ini, click the green run button in the upper left corner

### Update

Replace our requestResponseApp to Simu5G/simulations/NR/mec

Replace our VirtualisationInfrastructureManager to Simu5G/src/nodes/mec

Replace our MECOrchestrator to Simu5G/src/nodes/mec

### Run

Run Simu5G/simulations/NR/mec/requestResponseApp/omnetpp.ini

Run python DRL/main.py

### Results

Click Simu5G/simulations/NR/mec/requestResponseApp/results/MultiMec/MultiMec--0.sca to build MultiMec-.anf, analyze in IDE or export csv and analyze later (use MultiMecHost_delay.ue[0].app to filter to get the user key indicators).
