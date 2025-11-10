
package com.simulation;

public class Actuator {
    private final String name;

    public Actuator(String name) {
        this.name = name;
    }

    public double apply(double command) {
        // L'azione è una frazione del comando
        return command * 0.1;
    }
    
    public String getName() {
        return name;
    }
}
