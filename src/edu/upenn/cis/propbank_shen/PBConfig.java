package edu.upenn.cis.propbank_shen;

import java.io.File;


/*
   This class represents configuration information --
   particular filenames and directories of the propbank
   data set.
   @author Scott Cotton
 */
public class PBConfig {
    
    public static String PropBankFile() {
        return System.getProperty("PROPBANKFILE", "/Users/piffle/Desktop/spinalapi/spinalapi/prop-all.idx");
    }       

    public static String TreeBankDir() 
    { 
        return System.getProperty("TREEBANKDIR", "/Users/piffle/Desktop/spinalapi/spinalapi/ltagtb"); 
    }

    public static String FrameDir() 
    { 
        return System.getProperty("FRAMEDIR", "/Users/piffle/Desktop/spinalapi/spinalapi/frames");
    }
}
