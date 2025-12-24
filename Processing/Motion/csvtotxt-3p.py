# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:51:31 2022

@author: Martin Lee
"""


       
import pandas as pd


file="Convo002-local-euler.csv"
df = pd.read_csv(file,skiprows=1)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
frames=pd.read_csv(file,skiprows=1,usecols=['Frame']).iloc[-1].Frame

person1=df.iloc[0:frames]
person1.pop('Frame')
person2=df.iloc[frames+2:2*frames+2]
person2.pop('Frame')
person3=df.iloc[2*frames+4:3*frames+4]
person3.pop('Frame')
person4=df.iloc[3*frames+6:4*frames+6]
person4.pop('Frame')


temp = person1.pop('pelvis<t-X>')
person1=person1.join(temp)
temp = person1.pop('pelvis<t-Y>')
person1=person1.join(temp)
temp = person1.pop('pelvis<t-Z>')
person1=person1.join(temp)
temp = person1.pop('thorax<t-X>')
person1=person1.join(temp)
temp = person1.pop('thorax<t-Y>')
person1=person1.join(temp)
temp = person1.pop('thorax<t-Z>')
person1=person1.join(temp)
temp = person1.pop('head<t-X>')
person1=person1.join(temp)
temp = person1.pop('head<t-Y>')
person1=person1.join(temp)
temp = person1.pop('head<t-Z>')
person1=person1.join(temp)

temp = person2.pop('pelvis<t-X>')
person2=person2.join(temp)
temp = person2.pop('pelvis<t-Y>')
person2=person2.join(temp)
temp = person2.pop('pelvis<t-Z>')
person2=person2.join(temp)
temp = person2.pop('thorax<t-X>')
person2=person2.join(temp)
temp = person2.pop('thorax<t-Y>')
person2=person2.join(temp)
temp = person2.pop('thorax<t-Z>')
person2=person2.join(temp)
temp = person2.pop('head<t-X>')
person2=person2.join(temp)
temp = person2.pop('head<t-Y>')
person2=person2.join(temp)
temp = person2.pop('head<t-Z>')
person2=person2.join(temp)

temp = person3.pop('pelvis<t-X>')
person3=person3.join(temp)
temp = person3.pop('pelvis<t-Y>')
person3=person3.join(temp)
temp = person3.pop('pelvis<t-Z>')
person3=person3.join(temp)
temp = person3.pop('thorax<t-X>')
person3=person3.join(temp)
temp = person3.pop('thorax<t-Y>')
person3=person3.join(temp)
temp = person3.pop('thorax<t-Z>')
person3=person3.join(temp)
temp = person3.pop('head<t-X>')
person3=person3.join(temp)
temp = person3.pop('head<t-Y>')
person3=person3.join(temp)
temp = person3.pop('head<t-Z>')
person3=person3.join(temp)

temp = person4.pop('pelvis<t-X>')
person4=person4.join(temp)
temp = person4.pop('pelvis<t-Y>')
person4=person4.join(temp)
temp = person4.pop('pelvis<t-Z>')
person4=person4.join(temp)
temp = person4.pop('thorax<t-X>')
person4=person4.join(temp)
temp = person4.pop('thorax<t-Y>')
person4=person4.join(temp)
temp = person4.pop('thorax<t-Z>')
person4=person4.join(temp)
temp = person4.pop('head<t-X>')
person4=person4.join(temp)
temp = person4.pop('head<t-Y>')
person4=person4.join(temp)
temp = person4.pop('head<t-Z>')
person4=person4.join(temp)

person1.to_csv(r'person1.txt', sep=' ', index=False, header=False)
person2.to_csv(r'person2.txt', sep=' ', index=False, header=False)
person3.to_csv(r'person3.txt', sep=' ', index=False, header=False)
person4.to_csv(r'person4.txt', sep=' ', index=False, header=False)