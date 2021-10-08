package com.dailyQuestion.Octo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class findRepeatedDnaSequences187 {

    public static List<String> findRepeatedDnaSequences(String s) {
        HashMap<String, Integer> map = new HashMap<>();
        List<String> res = new ArrayList<>();

        for (int i = 0; i + 10 <= s.length() ; i++) {
            String str = s.substring(i,i+10);
            map.put(str,map.getOrDefault(str,0)+1);
            if(map.get(str) == 2) {
                res.add(str);
            }
        }

        return res;

    }
    //卡用例  最后一个超时了 暴力失败
    public static List<String> findRepeatedDnaSequences1(String s) {
        HashSet<String> set = new HashSet<>();

        for (int i = 0; i + 10 <= s.length() ; i++) {
            String str = s.substring(i,i+10);
            for (int j = i + 1; j + 10 <= s.length(); j++) {
                if(s.substring(j,j+10).equals(str)){
                    set.add(str);break;
                }
            }
        }

        List<String> res = new ArrayList<>();
        for (String s1 : set) {
            res.add(s1);
        }
        return res;

    }

    public static void main(String[] args) {
        String s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
        List<String> list = findRepeatedDnaSequences(s);
        for (String s1 : list) {
            System.out.println(s1);
        }
    }
}
