package at.technikum.ml.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

public class FileHelper {

    public static BufferedReader readDataFile(File file) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + file.getName());
        }

        return inputReader;
    }

}
