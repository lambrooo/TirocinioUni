
package com.simulation;

// Record per contenere i due valori di comando (simile a una tupla o a un piccolo oggetto dati)
record AgentCommands(double tempCommand, double loadCommand) {}

public class Agent {
    private final double tempSetpoint;
    private final double loadSetpoint;
    private final double pGain;

    public Agent(double tempSetpoint, double loadSetpoint, double pGain) {
        this.tempSetpoint = tempSetpoint;
        this.loadSetpoint = loadSetpoint;
        this.pGain = pGain;
    }

    public AgentCommands getCommands(double tempReading, double loadReading) {
        double tempError = this.tempSetpoint - tempReading;
        double loadError = this.loadSetpoint - loadReading;
        double tempCommand = this.pGain * tempError;
        double loadCommand = this.pGain * loadError;
        return new AgentCommands(tempCommand, loadCommand);
    }
}
