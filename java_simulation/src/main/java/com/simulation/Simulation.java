
package com.simulation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Simulation {

    public static void main(String[] args) {
        int nSteps = 1000;
        int anomalyInterval = 100;

        Environment env = new Environment(20.0, 30.0, 10.0);
        
        Map<String, Sensor> sensors = new HashMap<>();
        sensors.put("temperature", new Sensor("temperature", 0.5));
        sensors.put("motor_temperature", new Sensor("motor_temperature", 0.5));
        sensors.put("load", new Sensor("load", 0.2));

        Map<String, Actuator> actuators = new HashMap<>();
        actuators.put("temperature", new Actuator("temp_actuator"));
        actuators.put("load", new Actuator("load_actuator"));

        Agent agent = new Agent(60.0, 50.0, 0.2);

        List<String[]> logData = new ArrayList<>();
        logData.add(new String[] { "step", "true_temp", "sensed_temp", "true_motor_temp", "sensed_motor_temp", "true_load", "sensed_load", "temp_command", "load_command" });

        Random random = new Random();

        for (int step = 0; step < nSteps; step++) {
            // Lettura dai sensori
            double tempReading = sensors.get("temperature").read(env.getTemperature());
            double motorTempReading = sensors.get("motor_temperature").read(env.getMotorTemperature());
            double loadReading = sensors.get("load").read(env.getLoad());

            // Introduzione periodica di anomalie
            if (step > 0 && step % anomalyInterval == 0) {
                List<String> sensorKeys = new ArrayList<>(sensors.keySet());
                String sensorToAffect = sensorKeys.get(random.nextInt(sensorKeys.size()));
                String anomalyType = random.nextBoolean() ? "bias" : "outlier";
                double value = "bias".equals(anomalyType) ? 10.0 : 100.0;
                sensors.get(sensorToAffect).introduceAnomaly(anomalyType, value);
                System.out.printf("Step %d: Anomalia '%s' con valore %.1f su sensore '%s'%n", step, anomalyType, value, sensorToAffect);
            }

            // Calcolo comandi dell'agente
            AgentCommands commands = agent.getCommands(tempReading, loadReading);

            // Azioni degli attuatori
            double tempAction = actuators.get("temperature").apply(commands.tempCommand());
            double loadAction = actuators.get("load").apply(commands.loadCommand());

            // Aggiornamento dell'ambiente
            env.step(tempAction, loadAction);

            // Logging
            logData.add(new String[] {
                String.valueOf(step),
                String.valueOf(env.getTemperature()),
                String.valueOf(tempReading),
                String.valueOf(env.getMotorTemperature()),
                String.valueOf(motorTempReading),
                String.valueOf(env.getLoad()),
                String.valueOf(loadReading),
                String.valueOf(commands.tempCommand()),
                String.valueOf(commands.loadCommand())
            });
        }

        // Salvataggio in CSV
        saveToCsv(logData);
    }

    private static void saveToCsv(List<String[]> logData) {
        File csvFile = new File("simulation_log_java.csv");
        try (PrintWriter pw = new PrintWriter(csvFile)) {
            logData.stream()
              .map(data -> String.join(",", data))
              .forEach(pw::println);
            System.out.println("Dati di simulazione salvati in simulation_log_java.csv");
        } catch (FileNotFoundException e) {
            System.out.println("Errore durante il salvataggio del file CSV.");
            e.printStackTrace();
        }
    }
}
