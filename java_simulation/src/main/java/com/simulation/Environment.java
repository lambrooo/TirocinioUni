
package com.simulation;

import java.util.Random;

public class Environment {
    private double temperature;
    private double motorTemperature;
    private double load;
    private final double ambientTemp = 20.0;
    private final Random random = new Random();

    public Environment(double initialTemp, double initialMotorTemp, double initialLoad) {
        this.temperature = initialTemp;
        this.motorTemperature = initialMotorTemp;
        this.load = initialLoad;
    }

    public void step(double tempAction, double loadAction) {
        // La temperatura tende a tornare a quella ambiente
        this.temperature += 0.1 * (this.ambientTemp - this.temperature) + tempAction;
        // La temperatura del motore aumenta con il carico e si dissipa
        this.motorTemperature += 0.5 * this.load - 0.2 * (this.motorTemperature - this.temperature);
        // Il carico viene modificato dall'attuatore
        this.load += loadAction;
        // Aggiungiamo un po' di rumore di processo
        this.temperature += random.nextGaussian() * 0.1;
        this.motorTemperature += random.nextGaussian() * 0.1;
        this.load += random.nextGaussian() * 0.2;
    }

    // Getters
    public double getTemperature() { return temperature; }
    public double getMotorTemperature() { return motorTemperature; }
    public double getLoad() { return load; }
}
