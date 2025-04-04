import java.util.*;
import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class SimpleNetworkSliceSimulator {
    
    private static final Slice[] SLICES = {
        new Slice("MIoT", 95, 13),
        new Slice("eMBB", 15, 9),
        new Slice("HMTC", 1.2, 7),
        new Slice("URLLC", 1.3, 4),
        new Slice("V2X", 18, 11)
    };

    private static final TrafficProfile[] TRAFFIC_PROFILES = {
        new TrafficProfile("Video", 0.8, 2.0, 1),
        new TrafficProfile("Audio", 0.2, 0.8, 4),
        new TrafficProfile("IoT", 0.1, 0.3, 0),
        new TrafficProfile("WebData", 0.5, 1.5, 2),
        new TrafficProfile("ControlData", 0.3, 0.7, 3),
        new TrafficProfile("BestEffort", 0.1, 1.0, 2)
    };

    private List<UserEquipment> ues = new ArrayList<>();
    private double[] sliceLoads;
    private int[] ueCountPerSlice;
    private int currentStep = 0;
    private int totalUEs = 0;
    private int rejectedUEs = 0;
    private Random random = new Random(42);
    private double arrivalRate = 2.0;
    private SimplePythonAgent pythonAgent;

    public SimpleNetworkSliceSimulator() {
        sliceLoads = new double[SLICES.length];
        ueCountPerSlice = new int[SLICES.length];
    }

    public void initialize() {
        pythonAgent = new SimplePythonAgent();
        pythonAgent.start();
        
        System.out.println("Network Slice Simulator initialized successfully.");
        System.out.println("Connected to Python RL agent.");
    }

    public void runSimulation(int steps) {
        System.out.println("Starting network slice simulation for " + steps + " steps...");
        
        reset();
        
        for (int i = 0; i < steps; i++) {
            currentStep++;
            
            // Process new UE arrivals
            processArrivals();
            
            // Get the current observation/state
            double[] observation = getObservation();
            
            // Use the Python agent to select an action
            int action = pythonAgent.getAction(observation);
            
            // Apply the action (move UEs between slices)
            int sourceSlice = action / SLICES.length;
            int targetSlice = action % SLICES.length;
            processMovement(sourceSlice, targetSlice);
            
            // Update network state (UEs leaving, etc.)
            updateNetworkState();
            
            // Print status every 10 steps
            if (i % 10 == 0) {
                printStatus();
            }
        }
        
        System.out.println("\nSimulation complete!");
        printFinalStatistics();
    }
    
    private void reset() {
        currentStep = 0;
        ues.clear();
        sliceLoads = new double[SLICES.length];
        ueCountPerSlice = new int[SLICES.length];
        totalUEs = 0;
        rejectedUEs = 0;
    }
    
    private void processArrivals() {
        // Generate Poisson-distributed arrivals
        int numArrivals = getPoissonRandom(arrivalRate);
        
        // Every 10 steps, ensure we have all traffic types represented
        if (currentStep % 10 == 0) {
            for (int profileIdx = 0; profileIdx < TRAFFIC_PROFILES.length; profileIdx++) {
                TrafficProfile profile = TRAFFIC_PROFILES[profileIdx];
                double load = profile.getMinLoad() + random.nextDouble() * (profile.getMaxLoad() - profile.getMinLoad());
                addSpecificUE(profileIdx, load);
            }
        }
        
        // Process regular random arrivals
        for (int i = 0; i < numArrivals; i++) {
            addUE();
        }
    }
    
    private void addUE() {
        int profileIdx = random.nextInt(TRAFFIC_PROFILES.length);
        TrafficProfile profile = TRAFFIC_PROFILES[profileIdx];
        
        double load = profile.getMinLoad() + random.nextDouble() * (profile.getMaxLoad() - profile.getMinLoad());
        int preferredSlice = profile.getPreferredSlice();
        
        UserEquipment ue = new UserEquipment(profileIdx, load, preferredSlice, totalUEs);
        totalUEs++;
        
        // Try to allocate to preferred slice first
        if (sliceLoads[preferredSlice] + load <= SLICES[preferredSlice].getBandwidth()) {
            allocateUE(ue, preferredSlice);
        } else {
            // Try other slices
            boolean allocated = false;
            for (int slice = 0; slice < SLICES.length; slice++) {
                if (slice != preferredSlice && sliceLoads[slice] + load <= SLICES[slice].getBandwidth()) {
                    allocateUE(ue, slice);
                    allocated = true;
                    break;
                }
            }
            
            if (!allocated) {
                rejectedUEs++;
            }
        }
    }
    
    private void addSpecificUE(int profileIdx, double load) {
        TrafficProfile profile = TRAFFIC_PROFILES[profileIdx];
        int preferredSlice = profile.getPreferredSlice();
        
        UserEquipment ue = new UserEquipment(profileIdx, load, preferredSlice, totalUEs);
        totalUEs++;
        
        // Create a list of slice options with penalties
        List<SliceOption> sliceOptions = new ArrayList<>();
        sliceOptions.add(new SliceOption(preferredSlice, 0)); // No penalty for preferred slice
        
        // Add other slices with penalties
        for (int slice = 0; slice < SLICES.length; slice++) {
            if (slice != preferredSlice) {
                sliceOptions.add(new SliceOption(slice, 0.1));
            }
        }
        
        // Sort by relative load + penalty
        sliceOptions.sort(Comparator.comparingDouble(
            option -> (sliceLoads[option.getSliceIdx()] / SLICES[option.getSliceIdx()].getBandwidth()) + option.getPenalty()
        ));
        
        // Try to allocate to the best slice
        boolean allocated = false;
        for (SliceOption option : sliceOptions) {
            int slice = option.getSliceIdx();
            if (sliceLoads[slice] + load <= SLICES[slice].getBandwidth()) {
                allocateUE(ue, slice);
                allocated = true;
                break;
            }
        }
        
        if (!allocated) {
            rejectedUEs++;
        }
    }
    
    private void allocateUE(UserEquipment ue, int sliceIdx) {
        ue.setAllocatedSlice(sliceIdx);
        ues.add(ue);
        sliceLoads[sliceIdx] += ue.getLoad();
        ueCountPerSlice[sliceIdx]++;
    }
    
    private void processMovement(int sourceSlice, int targetSlice) {
        if (sourceSlice == targetSlice || sourceSlice >= SLICES.length || targetSlice >= SLICES.length || 
            ueCountPerSlice[sourceSlice] == 0) {
            return;
        }
        
        // Find UEs to move
        List<UserEquipment> uesToMove = ues.stream()
            .filter(ue -> ue.getAllocatedSlice() == sourceSlice)
            .collect(Collectors.toList());
        
        if (uesToMove.isEmpty()) {
            return;
        }
        
        double movedLoad = uesToMove.stream().mapToDouble(UserEquipment::getLoad).sum();
        
        // Check if target slice can accommodate all UEs
        if (sliceLoads[targetSlice] + movedLoad <= SLICES[targetSlice].getBandwidth()) {
            // Move UEs
            for (UserEquipment ue : uesToMove) {
                ue.setAllocatedSlice(targetSlice);
            }
            
            sliceLoads[sourceSlice] -= movedLoad;
            sliceLoads[targetSlice] += movedLoad;
            
            ueCountPerSlice[sourceSlice] -= uesToMove.size();
            ueCountPerSlice[targetSlice] += uesToMove.size();
            
            System.out.println("Moved " + uesToMove.size() + " UEs from " + 
                              SLICES[sourceSlice].getName() + " to " + 
                              SLICES[targetSlice].getName());
        }
    }
    
    private void updateNetworkState() {
        // Process UEs leaving the network
        Iterator<UserEquipment> iterator = ues.iterator();
        while (iterator.hasNext()) {
            UserEquipment ue = iterator.next();
            ue.incrementTimeInNetwork();
            
            // Chance of UE leaving increases with time
            double leaveProb = Math.min(0.05, 0.001 * ue.getTimeInNetwork());
            if (random.nextDouble() < leaveProb) {
                int sliceIdx = ue.getAllocatedSlice();
                sliceLoads[sliceIdx] -= ue.getLoad();
                ueCountPerSlice[sliceIdx]--;
                iterator.remove();
            }
        }
    }
    
    private double[] getObservation() {
        // Create observation vector similar to Python environment
        int obsSize = SLICES.length * 2 + SLICES.length * TRAFFIC_PROFILES.length;
        double[] observation = new double[obsSize];
        
        // UE count per slice (normalized)
        int maxUECount = Math.max(1, Arrays.stream(ueCountPerSlice).max().orElse(1));
        for (int i = 0; i < SLICES.length; i++) {
            observation[i] = (double) ueCountPerSlice[i] / maxUECount;
        }
        
        // UE types per slice (simplified)
        int offset = SLICES.length;
        int[][] ueTypesPerSlice = new int[SLICES.length][TRAFFIC_PROFILES.length];
        for (UserEquipment ue : ues) {
            int slice = ue.getAllocatedSlice();
            int profile = ue.getProfile();
            ueTypesPerSlice[slice][profile]++;
        }
        
        for (int i = 0; i < SLICES.length; i++) {
            for (int j = 0; j < TRAFFIC_PROFILES.length; j++) {
                int idx = offset + i * TRAFFIC_PROFILES.length + j;
                observation[idx] = (double) ueTypesPerSlice[i][j] / Math.max(1, ueCountPerSlice[i]);
            }
        }
        
        // Slice loads (normalized)
        offset = SLICES.length + (SLICES.length * TRAFFIC_PROFILES.length);
        for (int i = 0; i < SLICES.length; i++) {
            observation[offset + i] = sliceLoads[i] / SLICES[i].getBandwidth();
        }
        
        return observation;
    }
    
    private void printStatus() {
        System.out.println("\nStep " + currentStep + " Status:");
        System.out.println("Total UEs: " + totalUEs + ", Active UEs: " + ues.size() + 
                ", Rejected: " + rejectedUEs);
        
        System.out.println("Slice Loads:");
        for (int i = 0; i < SLICES.length; i++) {
            double utilizationPct = (sliceLoads[i] / SLICES[i].getBandwidth()) * 100;
            System.out.printf("  %s: %.2f Mbps (%.1f%% utilized, %d UEs)\n", 
                    SLICES[i].getName(), sliceLoads[i], utilizationPct, ueCountPerSlice[i]);
        }
    }
    
    private void printFinalStatistics() {
        System.out.println("\n=== Final Statistics ===");
        System.out.println("Total UEs processed: " + totalUEs);
        System.out.println("Rejected UEs: " + rejectedUEs + 
                " (" + String.format("%.1f%%", (double)rejectedUEs/totalUEs*100) + ")");
        
        System.out.println("\nFinal Slice Utilization:");
        for (int i = 0; i < SLICES.length; i++) {
            double utilizationPct = (sliceLoads[i] / SLICES[i].getBandwidth()) * 100;
            System.out.printf("  %s: %.2f Mbps (%.1f%% utilized)\n", 
                    SLICES[i].getName(), sliceLoads[i], utilizationPct);
        }
    }
    
    private int getPoissonRandom(double mean) {
        double L = Math.exp(-mean);
        double p = 1.0;
        int k = 0;
        
        do {
            k++;
            p *= random.nextDouble();
        } while (p > L);
        
        return k - 1;
    }
    
    public void shutdown() {
        if (pythonAgent != null) {
            pythonAgent.stop();
        }
        System.out.println("Network Slice Simulator shut down.");
    }
    
    public static void main(String[] args) {
        SimpleNetworkSliceSimulator simulator = new SimpleNetworkSliceSimulator();
        
        try {
            simulator.initialize();
            simulator.runSimulation(200);
        } finally {
            simulator.shutdown();
        }
    }
    
    // Support classes
    static class Slice {
        private String name;
        private double bandwidth;
        private int latency;
        
        public Slice(String name, double bandwidth, int latency) {
            this.name = name;
            this.bandwidth = bandwidth;
            this.latency = latency;
        }
        
        public String getName() { return name; }
        public double getBandwidth() { return bandwidth; }
        public int getLatency() { return latency; }
    }
    
    static class TrafficProfile {
        private String name;
        private double minLoad;
        private double maxLoad;
        private int preferredSlice;
        
        public TrafficProfile(String name, double minLoad, double maxLoad, int preferredSlice) {
            this.name = name;
            this.minLoad = minLoad;
            this.maxLoad = maxLoad;
            this.preferredSlice = preferredSlice;
        }
        
        public String getName() { return name; }
        public double getMinLoad() { return minLoad; }
        public double getMaxLoad() { return maxLoad; }
        public int getPreferredSlice() { return preferredSlice; }
    }
    
    static class UserEquipment {
        private int profile;
        private double load;
        private int preferredSlice;
        private int allocatedSlice;
        private int timeInNetwork;
        private int id;
        
        public UserEquipment(int profile, double load, int preferredSlice, int id) {
            this.profile = profile;
            this.load = load;
            this.preferredSlice = preferredSlice;
            this.id = id;
            this.timeInNetwork = 0;
            this.allocatedSlice = -1;
        }
        
        public int getProfile() { return profile; }
        public double getLoad() { return load; }
        public int getPreferredSlice() { return preferredSlice; }
        public int getAllocatedSlice() { return allocatedSlice; }
        public void setAllocatedSlice(int slice) { this.allocatedSlice = slice; }
        public int getTimeInNetwork() { return timeInNetwork; }
        public void incrementTimeInNetwork() { this.timeInNetwork++; }
        public int getId() { return id; }
    }
    
    static class SliceOption {
        private int sliceIdx;
        private double penalty;
        
        public SliceOption(int sliceIdx, double penalty) {
            this.sliceIdx = sliceIdx;
            this.penalty = penalty;
        }
        
        public int getSliceIdx() { return sliceIdx; }
        public double getPenalty() { return penalty; }
    }
}

/**
 * Simple Python agent interface using process-based communication
 */
class SimplePythonAgent {
    private Process pythonProcess;
    private BufferedWriter processInput;
    private BufferedReader processOutput;
    
    public void start() {
        try {
            // Create a simple Python script for the agent
            String scriptPath = createPythonScript();
            
            // Start the Python process
            ProcessBuilder pb = new ProcessBuilder("python", scriptPath);
            pb.redirectErrorStream(true);
            
            System.out.println("Starting Python agent process...");
            pythonProcess = pb.start();
            
            // Set up communication channels
            processInput = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
            processOutput = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
            
            // Wait for startup message
            String startupMsg = processOutput.readLine();
            System.out.println("Python agent: " + startupMsg);
            
        } catch (Exception e) {
            System.err.println("Failed to start Python agent: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public int getAction(double[] observation) {
        try {
            // Convert observation to JSON-like string
            StringBuilder obsString = new StringBuilder("[");
            for (int i = 0; i < observation.length; i++) {
                obsString.append(observation[i]);
                if (i < observation.length - 1) {
                    obsString.append(", ");
                }
            }
            obsString.append("]");
            
            // Send observation to Python process
            processInput.write(obsString.toString());
            processInput.newLine();
            processInput.flush();
            
            // Read action from Python process
            String actionStr = processOutput.readLine();
            return Integer.parseInt(actionStr.trim());
            
        } catch (Exception e) {
            System.err.println("Error getting action from Python agent: " + e.getMessage());
            // Return a default action (move from slice 0 to 1)
            return 1;
        }
    }
    
    public void stop() {
        try {
            if (processInput != null) {
                processInput.write("EXIT");
                processInput.newLine();
                processInput.flush();
                processInput.close();
            }
            
            if (processOutput != null) {
                processOutput.close();
            }
            
            if (pythonProcess != null) {
                pythonProcess.waitFor(2, TimeUnit.SECONDS);
                pythonProcess.destroy();
            }
        } catch (Exception e) {
            System.err.println("Error shutting down Python agent: " + e.getMessage());
        }
    }
    
    private String createPythonScript() throws IOException {
        // Create a temporary file for the Python script
        File tempScript = File.createTempFile("slice_agent_", ".py");
        tempScript.deleteOnExit();
        
        try (FileWriter writer = new FileWriter(tempScript)) {
            writer.write(
                "import sys\n" +
                "import os\n" +
                "import torch\n" +
                "import numpy as np\n" +
                "\n" +
                "# Get the path to the ML project\n" +
                "project_path = r'" + System.getProperty("user.dir") + "'\n" +
                "sys.path.append(project_path)\n" +
                "\n" +
                "# Import the agents from main.py if possible\n" +
                "try:\n" +
                "    from main import SACAgent, DQNAgent, PPOAgent, NetworkSlicingEnv\n" +
                "    \n" +
                "    # Initialize environment to get dimensions\n" +
                "    env = NetworkSlicingEnv(arrival_rate=2)\n" +
                "    state_dim = env.observation_space.shape[0]\n" +
                "    action_dim = env.action_space.n\n" +
                "    \n" +
                "    # Try to load a trained agent\n" +
                "    if os.path.exists('models/sac_final.pth'):\n" +
                "        print('Loading SAC agent...')\n" +
                "        agent = SACAgent(state_dim, action_dim)\n" +
                "        agent.actor.load_state_dict(torch.load('models/sac_final.pth'))\n" +
                "    elif os.path.exists('models/ppo_final.pth'):\n" +
                "        print('Loading PPO agent...')\n" +
                "        agent = PPOAgent(state_dim, action_dim)\n" +
                "        agent.policy.load_state_dict(torch.load('models/ppo_final.pth'))\n" +
                "    elif os.path.exists('models/dqn_final.pth'):\n" +
                "        print('Loading DQN agent...')\n" +
                "        agent = DQNAgent(state_dim, action_dim)\n" +
                "        agent.q_network.load_state_dict(torch.load('models/dqn_final.pth'))\n" +
                "    else:\n" +
                "        print('No trained models found. Using random actions')\n" +
                "        agent = None\n" +
                "except ImportError as e:\n" +
                "    print(f'Failed to import agents: {e}')\n" +
                "    agent = None\n" +
                "\n" +
                "print('Python agent ready!')\n" +
                "sys.stdout.flush()\n" +
                "\n" +
                "# Main loop to receive observations and return actions\n" +
                "while True:\n" +
                "    try:\n" +
                "        # Read input from Java\n" +
                "        input_line = input().strip()\n" +
                "        \n" +
                "        # Check for exit command\n" +
                "        if input_line == 'EXIT':\n" +
                "            break\n" +
                "        \n" +
                "        # Parse observation array\n" +
                "        input_line = input_line.strip('[]')\n" +
                "        observation = np.array([float(x) for x in input_line.split(',')], dtype=np.float32)\n" +
                "        \n" +
                "        # Get action from agent\n" +
                "        if agent is not None:\n" +
                "            action = agent.select_action(observation, evaluation=True)\n" +
                "        else:\n" +
                "            # Random action if no agent is available\n" +
                "            action = np.random.randint(0, 25)  # Assuming 5x5 slices\n" +
                "        \n" +
                "        # Return action\n" +
                "        print(int(action))\n" +
                "        sys.stdout.flush()\n" +
                "        \n" +
                "    except Exception as e:\n" +
                "        print(f'Error in Python agent: {e}')\n" +
                "        print(0)  # Default action\n" +
                "        sys.stdout.flush()\n"
            );
        }
        
        return tempScript.getAbsolutePath();
    }
}