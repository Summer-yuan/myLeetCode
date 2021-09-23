package com.dailyQuestion.Sept;

import java.util.Arrays;

public class findNumberOfLIS673 {
    public int findNumberOfLIS(int[] nums) {
        if (nums.length <= 1) {
            return nums.length;
        }
        int len = nums.length;
        int[] dp = new int[len];
        int[] num = new int[len];
        Arrays.fill(dp,1);
        Arrays.fill(num,1);
        int longLength = 0;

        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if(nums[j] < nums[i]){

                    if(dp[j] + 1 == dp[i]){
                        num[i] += num[j];
                    }else if(dp[j] + 1 > dp[i]){
                        dp[i] = dp[j] + 1;
                        num[i] = num[j];
                    }
                }
                longLength = Math.max(longLength,dp[i]);
            }
        }

        int res = 0;
        for (int i = 0; i < len; i++) {
            if(dp[i] == longLength) {
                res += num[i];
            }
        }
        return res;

    }
}
