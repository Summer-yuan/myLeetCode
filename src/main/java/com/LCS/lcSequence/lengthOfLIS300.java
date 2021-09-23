package com.LCS.lcSequence;

import java.util.Arrays;

public class lengthOfLIS300 {
    public static int lengthOfLIS(int[] nums) {
        int len = nums.length;
        int res = 0;
        int[] dp = new int[len];
        Arrays.fill(dp,1);
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if(nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i],dp[j]+1);
                }
            }
            res = Math.max(res,dp[i]);
        }

        return res;
    }

    public static void main(String[] args) {
        int[] arr = {0,1,0,3,2,3};
        System.out.println(lengthOfLIS(arr));
    }
}
