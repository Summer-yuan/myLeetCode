package com.nowCoder.shopeeAdvance2022;

import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

//https://www.nowcoder.com/question/next?pid=32331975&qid=2049913&tid=48327626
public class Solution {

    public String char_and_num_return (String text_source) {
        StringBuilder sb = new StringBuilder();
        String[] split = text_source.split("[ ]+");
        PriorityQueue pq = new PriorityQueue<String>(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (int) (Long.parseLong(o1)-Long.parseLong(o2));
            }
        });

        for (String s : split) {
            if(s.charAt(0) >= '0' && s.charAt(0) <= '9'){
                pq.offer(s);
            }else{
                sb.append(s+" ");
            }
        }

        while(!pq.isEmpty()){
            sb.append(pq.poll()+" ");
        }

        return sb.toString().trim();


    }

    public static int cost (int[] array) {
        int len = array.length;
        int[] cost = new int[len];
        Arrays.fill(cost,1);
        for (int i = 1; i < array.length; i++) {
            if(array[i] > array[i-1] && cost[i] <= cost[i-1]){
                cost[i] = cost[i-1] + 1;
            }
        }
        for (int i = array.length - 2; i >= 0; i--) {
            if(array[i] > array[i+1] && cost[i] <= cost[i+1]){
                cost[i] = cost[i+1] + 1;
            }
        }
        int total = 0;
        for (int i : cost) {
            total += i;
            System.out.print(i+" ");
        }
        return total;
    }

    public static void main(String[] args) {
        int[] arr = {1,2,3,0,5,2,5,9};
        System.out.println(cost(arr));
    }
}
