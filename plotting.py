import matplotlib.pyplot as plt


# Close up behavior of learning proccess
# each algorithm run on 
fictitious_play = [1.552642, 2.0, 1.5686450000000003, 1.751345, 1.626857, 1.7857210000000001, 2.0, 1.6120450000000002, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003, 1.5686450000000003]
epsilon_greedy = [1.555621, 1.580441, 1.5840999999999998, 1.5773089999999999, 1.5840999999999998, 1.583645, 1.585397, 1.5812089999999999, 1.5857539999999999, 1.58357, 1.5673690000000002, 1.5785360000000002, 1.579641, 1.583785, 1.579234, 1.567825, 1.581469, 1.576058, 1.5840520000000002, 1.588625, 1.580018, 1.57209, 1.572593, 1.579641, 1.5844610000000001, 1.56781, 1.5706689999999999, 1.5732040000000003, 1.5754290000000002, 1.572945, 1.5789849999999999, 1.57225, 1.5689050000000002, 1.576141, 1.5785140000000002, 1.57306, 1.571514, 1.5746980000000002, 1.579913, 1.5651610000000002, 1.590493, 1.5833039999999998, 1.574897, 1.579865, 1.5921329999999998, 1.5763379999999998, 1.5832819999999999, 1.582013, 1.5707200000000001, 1.5837439999999998, 1.581217, 1.57769, 1.569745, 1.582125, 1.5813490000000001, 1.5903249999999998, 1.5896089999999998, 1.5879250000000003, 1.5826600000000002, 1.580581, 1.582025, 1.581616, 1.5817359999999998, 1.57949, 1.580128, 1.5921699999999999, 1.5699700000000003, 1.585297, 1.585705, 1.58969, 1.5774449999999998, 1.5879250000000003, 1.571485, 1.586797, 1.5828489999999997, 1.5833599999999999, 1.580665, 1.577032, 1.5842530000000001, 1.5727119999999999, 1.5856999999999999, 1.5901489999999998, 1.5847250000000002, 1.5765250000000002, 1.58609, 1.583785, 1.5754970000000001, 1.581706, 1.5818960000000002, 1.579444, 1.583045, 1.570816, 1.5965049999999998, 1.5825799999999999, 1.5820169999999998, 1.5718339999999997, 1.5737599999999998, 1.5745700000000002]
UCB1 = [1.554785, 2.0, 1.500032, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.750016, 1.750016, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 1.750016, 1.750016, 1.500032, 2.0, 1.500032, 1.500032, 1.500032, 1.750016, 1.750016, 1.500032, 2.0, 1.500032, 1.500032, 1.500032]
thompson = [1.571444, 1.6101599999999998, 1.6551369999999999, 1.709317, 1.795993, 1.86297, 1.895173, 1.9014159999999998, 1.9389969999999999, 1.958932, 1.951252, 1.9685139999999999, 1.9811929999999998, 1.962722, 1.979233, 1.989061, 1.9880799999999998, 1.983185, 1.987125, 1.987089, 1.9831569999999998, 1.991041, 1.993025, 1.9900579999999999, 1.9950130000000001, 1.992034, 1.99601, 1.99402, 1.9920320000000002, 1.9990010000000002, 1.996008, 1.996008, 1.9990010000000002, 1.99601, 1.997005, 1.9950130000000001, 1.9970089999999998, 1.995017, 1.997005, 1.998004, 1.9990010000000002, 1.998004, 2.0, 1.9950130000000001, 1.9990010000000002, 1.9990010000000002, 2.0, 1.99601, 1.9990010000000002, 1.998002, 1.998004, 1.996008, 1.9990010000000002, 1.998004, 1.9990010000000002, 1.998004, 1.9990010000000002, 2.0, 1.9990010000000002, 2.0, 1.998004, 2.0, 1.9970089999999998, 2.0, 2.0, 1.997005, 1.9990010000000002, 2.0, 2.0, 1.998002, 2.0, 2.0, 1.9990010000000002, 2.0, 1.998004, 2.0, 2.0, 1.9990010000000002, 2.0, 2.0, 2.0, 2.0, 1.9990010000000002, 1.9990010000000002, 2.0, 2.0, 1.9990010000000002, 1.998004, 1.9990010000000002, 1.9990010000000002, 2.0, 2.0, 2.0, 1.9990010000000002, 2.0, 1.9990010000000002, 2.0, 2.0, 2.0, 2.0]
x =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]



plt.plot(avg_agent_cost_UCB1, color='k')
plt.title("UCB1, 4,000 agents, with a highway")
plt.xlabel("rounds")
plt.ylabel("avg cost per agent")
plt.show()