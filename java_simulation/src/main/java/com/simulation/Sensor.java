
package com.simulation;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Sensor {
    private final String name;
    private final double noiseStdDev;
    private final Random random = new Random();
    private Map<String, Object> anomaly = null;

    public Sensor(String name, double noiseStdDev) {
        this.name = name;
        this.noiseStdDev = noiseStdDev;
    }

    public double read(double trueValue) {
        double reading = trueValue + random.nextGaussian() * this.noiseStdDev;
        if (this.anomaly != null) {
            String type = (String) this.anomaly.get("type");
            double value = (double) this.anomaly.get("value");
            if ("bias".equals(type)) {
                reading += value;
            } else if ("outlier".equals(type)) {
                reading = value;
            }
            this.anomaly = null; // L'anomalia è istantanea
        }
        return reading;
    }

    public void introduceAnomaly(String anomalyType, double value) {
        this.anomaly = new HashMap<>();
        this.anomaly.put("type", anomalyType);
        this.anomaly.put("value", value);
    }
    
    public String getName() {
        return name;
    }
}
