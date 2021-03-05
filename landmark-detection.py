import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import pygame, sys
from pygame.locals import *
pygame.init()

pygame.display.set_caption("Landmark Detection")
screen = pygame.display.set_mode((700,500),0,32)

def draw_rectangle(surf, color, x1,y1,x2,y2):
    pygame.draw.line(surf, color, (x1,y1), (x1,y2))
    pygame.draw.line(surf, color, (x1,y1), (x2,y1))
    pygame.draw.line(surf, color, (x2,y2), (x1,y2))
    pygame.draw.line(surf, color, (x2,y2), (x2,y1))

def blink_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)
display = np.zeros((500,700,3), np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

positions = []
rect_pos = []
blinks = 0
eyes_open = True
end_connections = {67:60, 42:47, 36:41}

font = pygame.font.SysFont(None, 24)


while True:
    screen.fill((0,0,25))
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    found_face = False
    for face in faces:	
        found_face = True
        display = np.zeros((500,700,3), np.uint8)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        rect_pos = [x1,y1,x2,y2]
        #pygame.draw.rect(screen, (255,255,255), pygame.Rect(x1,y1,x2-x1,y2-y1))
        
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        landmarks = predictor(gray, face)
        
        positions = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            positions.append((x,y))
        
        lefteye = positions[36:42]
        righteye = positions[42:48]
        left_ratio = blink_ratio(lefteye)
        right_ratio = blink_ratio(righteye)
        ear = (left_ratio + right_ratio) / 2.0
        if ear <= 0.21:
            if eyes_open:
                blinks += 1
                eyes_open = False
        else:
            eyes_open = True

    for index,pos in enumerate(positions):
        if index != 0 and index != 17 and index != 22 and index != 27 and index != 48 and index != 36 and index != 42:
            pygame.draw.line(screen, (200,160,160), pos, old_pos, 4)
        old_pos = pos
        #screen.blit(font.render(str(index), True, (255,255,255)),pos)

    for index in list(end_connections.keys()):
        if positions:
            pygame.draw.line(screen, (200,160,160), positions[index], positions[end_connections[index]], 4)
        #pygame.draw.circle(screen, (200,160,160), (pos[0],pos[1]), 4)
    #if rect_pos: 
    #    draw_rectangle(screen, (255,255,255), x1,y1,x2,y2)
    screen.blit(font.render(f"Found Face: {found_face}", True, (255,255,255)), (0,0))
    screen.blit(font.render(f"Blinks: {blinks}", True, (255,255,255)), (0,32))
    screen.blit(font.render(f"Eyes Open: {eyes_open}", True, (255,255,255)), (0,64))
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
